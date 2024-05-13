import sys
import torch
import io
from PIL import Image
from torchvision import transforms
from diffusers import StableDiffusionInpaintPipeline, EulerDiscreteScheduler

# 추가된 경로 (필요에 따라 수정 가능)
sys.path.append('/home/etri/workspace/minji/system/clipseg_repo')
from models.clipseg import CLIPDensePredT

# 재난 유형에 따른 영어 단어 매핑
disaster_type_to_english = {
    '지진': 'earthquake',
    '지반침하': 'land subsidence',
    '싱크홀': 'sinkhole',
    '토석류': 'mudslide',
    '홍수': 'flood',
    '폭풍해일': 'storm surge',
    '산불': 'wildfire',
    '화재': 'fire',
    '폭발사고': 'explosion',
    '산사태': 'landslide'
}

# Segment prompt 설정 함수
def get_segment_prompt(disaster_type):
    mountain_related = ['산불', '산사태']
    building_related = ['화재', '폭발사고']
    ground_related = ['지진', '지반침하', '싱크홀', '토석류']
    water_related = ['홍수', '폭풍해일']

    if disaster_type in mountain_related:
        return 'mountain'
    elif disaster_type in building_related:
        return 'building'
    elif disaster_type in ground_related:
        return 'ground'
    elif disaster_type in water_related:
        return 'ocean'
    return 'unknown'

# 마스크 확장/축소 및 하단 절반/3/4 선택 함수
def adjust_mask_based_on_alert_intensity(processed_mask, alert_intensity, expand=True):
    if expand:
        kernel_size = 20
        if alert_intensity == '주의보':
            kernel_size = 40
        elif alert_intensity == '경보':
            kernel_size = 60
        adjusted_mask = torch.nn.functional.max_pool2d(processed_mask.unsqueeze(0).unsqueeze(0), kernel_size=kernel_size, stride=1, padding=kernel_size//2).squeeze()
    else:
        kernel_size = 20
        if alert_intensity == '주의보':
            kernel_size = 10
        elif alert_intensity == '경보':
            kernel_size = 5
        adjusted_mask = torch.nn.functional.min_pool2d(processed_mask.unsqueeze(0).unsqueeze(0), kernel_size=kernel_size, stride=1, padding=kernel_size//2).squeeze()

        # 이미지의 크기를 가져옵니다.
        _, height = adjusted_mask.shape

        # 마스크의 하단 절반 또는 3/4을 선택합니다.
        if alert_intensity == '주의보':
            # 하단 절반 선택
            half_mask = torch.zeros_like(adjusted_mask)
            half_mask[height//2:] = adjusted_mask[height//2:]
            adjusted_mask = half_mask
        elif alert_intensity == '경보':
            # 하단 3/4 선택
            quarter_mask = torch.zeros_like(adjusted_mask)
            quarter_mask[height//4:] = adjusted_mask[height//4:]
            adjusted_mask = quarter_mask

    return (adjusted_mask - adjusted_mask.min()) / (adjusted_mask.max() - adjusted_mask.min())

# 인페인팅 함수
def apply_inpainting(image, disaster_type, alert_intensity):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 이미지 데이터 타입 확인 및 변환
    if isinstance(image, io.BytesIO):
        image.seek(0)  # 스트림의 시작 부분으로 포인터 이동
        image = Image.open(image).convert('RGB')  # BytesIO를 PIL Image 객체로 변환
    elif isinstance(image, str):
        image = Image.open(image).convert('RGB')  # 파일 경로에서 이미지 로드 및 변환

    tensor_image = transforms.ToTensor()(image).unsqueeze(0).to(device)

    # 모델 설정
    model_path = '/home/etri/workspace/minji/system/clipseg_weights/clipseg_weights/rd64-uni.pth'
    model = CLIPDensePredT(version='ViT-B/16', reduce_dim=64).to(device)
    model.eval()
    model.load_state_dict(torch.load(model_path, map_location=device), strict=False)

    # Stable Diffusion 인페인팅 파이프라인 설정
    model_dir = "stabilityai/stable-diffusion-2-inpainting"
    scheduler = EulerDiscreteScheduler.from_pretrained(model_dir, subfolder="scheduler").to(device)
    pipe = StableDiffusionInpaintPipeline.from_pretrained(model_dir, scheduler=scheduler, revision="fp16", torch_dtype=torch.float32).to(device)

    # ClipSeg를 사용하여 마스크 생성
    segment_prompt = get_segment_prompt(disaster_type)
    with torch.no_grad():
        preds = model(tensor_image, [segment_prompt])[0]
    processed_mask = torch.special.ndtr(preds[0][0]).to(device)

    # 마스크 확장 또는 축소 적용
    expand = disaster_type not in ['산불', '산사태', '화재', '폭발사고']
    expanded_mask = adjust_mask_based_on_alert_intensity(processed_mask, alert_intensity, expand=expand)
    stable_diffusion_mask = transforms.ToPILImage()(expanded_mask.cpu())

    # 인페인팅 프롬프트 설정
    english_disaster_type = disaster_type_to_english.get(disaster_type, 'disaster')
    inpainting_prompt = f"{english_disaster_type} has occurred"

    # 이미지 인페인팅을 수행합니다.
    generator = torch.Generator(device=device).manual_seed(77)
    result_image = pipe(prompt=inpainting_prompt, guidance_scale=7.5, num_inference_steps=60, generator=generator, image=tensor_image, mask_image=stable_diffusion_mask).images[0]

    return result_image
