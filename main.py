import sys
from .chatgpt import parse_disaster_alert
from .rainy import create_rain_effect
from .snowy import create_snow_effect
from .background_inpainting import background_inpainting
from .inpainting import apply_inpainting
import pickle

def main(parsed_alert, image_path):


    disaster_types = parsed_alert['재난 종류'].split(", ")  # '호우, 홍수'와 같은 입력을 리스트로 변환
    alert_intensity = parsed_alert.get('재난 강도', None)

    global_disasters = ['태풍', '호우', '대설', '우박']
    surface_disasters = ['지진', '지반침하', '싱크홀', '토석류', '홍수', '폭풍해일']
    structure_focused_disasters = ['산불', '화재', '폭발사고', '산사태']  # 구조에 중점을 둔 재난 유형 추가

    # 전역 규모 재난과 지표면 재난을 구분하기 위한 플래그
    has_global_disaster = any(disaster_type in global_disasters for disaster_type in disaster_types)
    has_surface_disaster = any(disaster_type in surface_disasters for disaster_type in disaster_types)
    has_structure_focused_disaster = any(disaster_type in structure_focused_disasters for disaster_type in disaster_types)


    # Step 1: 전역 규모 재난 여부 확인
    if has_global_disaster:
        # Step 2: 전역 규모 재난이 있을 경우, 배경 전처리 진행
        print("전역 규모 재난에 대한 배경 전처리를 적용 중...")
        final_image = background_inpainting(image_path)

    # Step 2와 3: 추가적인 지표면 재난 처리
    if has_surface_disaster:
        # 지표면 재난에 대한 처리가 필요한 경우
        for disaster_type in disaster_types:
            if disaster_type in surface_disasters:
                print(f"{disaster_type}에 대한 인페인팅을 적용 중...")
                final_image = apply_inpainting(final_image if 'final_image' in locals() else image_path, disaster_type, alert_intensity)

    # Step 2와 3: 추가적인 지표면 재난 처리
    if has_structure_focused_disaster:
        # 지표면 재난에 대한 처리가 필요한 경우
        for disaster_type in disaster_types:
            if disaster_type in structure_focused_disasters:
                print(f"{disaster_type}에 대한 인페인팅을 적용 중...")
                final_image = apply_inpainting(final_image if 'final_image' in locals() else image_path, disaster_type, alert_intensity)

    # 전역 규모 재난의 추가적인 이미지 변환
    if has_global_disaster:
        for disaster_type in disaster_types:
            if disaster_type == '호우':
                print("비 효과를 적용 중...")
                final_image = create_rain_effect(final_image if 'final_image' in locals() else image_path)
            elif disaster_type == '대설':
                print("눈 효과를 적용 중...")
                final_image = create_snow_effect(final_image if 'final_image' in locals() else image_path)

    # 최종 이미지를 표시하거나 저장합니다.
    if 'final_image' in locals():
        final_image.save('static/disaster/disaster.png')  # 최종 이미지를 표시합니다.
    else:
        print("적용된 재난 효과가 없습니다.")


def parsed_disaster(alert_text):


    parsed_alert = parse_disaster_alert(alert_text)
    print(parsed_alert)
    with open ("data.pickle", "wb") as fw:
        pickle.dump(parsed_alert, fw)

    # with open("data.pickle", "rb") as fr : 
        # parsed_alert = pickle.load(fr)

    return parsed_alert
