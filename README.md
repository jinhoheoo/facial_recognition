# 얼굴 표정 인식

객체탐지를 통해 얼굴을 인식한 이후 표정을 인식하는 프로젝트 입니다. 

수행기간: 2024.03.20~2024.03.22

목표: 쉽게 구현 할 수 있는 Haar Cascade를 통한 Object Detection과 Open Model Zoo에서 제공하는 디바이스에 맞게 최적화하여 객체 인식을 하는 코드를 비교하며 어떤게 최종 프로젝트에서 이용할 Object Detection 모델을 선택합니다.

# 제작한 UI 및 실행화면

Haar Cascade를 이용한 객체탐지

![image](https://github.com/jinhoheoo/food_recommendation_project/assets/153490852/8a4400a7-a7b2-4997-81ae-e2ca73dda604)

Mobilenet SSD을 이용한 객체탐지

![image](https://github.com/jinhoheoo/food_recommendation_project/assets/153490852/83e66033-d55d-4501-9326-59708db4c7df)


# 설계 결과

1. Haar Cascade는 모델의 크기가 작고 빠르게 동작한다는 장점이 있지만 객체 인식률이 떨어집니다.

2. Mobilenet SSD 또한 비슷한 장점이 있지만 해당 프로젝트에서 이용하기 위해서는 intel에서 제공하는 otx를 이용하여 fine-tuning하여 얼굴만을 인식하게 하는 과정이 필요합니다. 현재의 모델은 다양한 객체를 인지하기 때문에 얼굴만을 인지하기 위한 모델을 만들어야 합니다.

3. 해당 프로젝트에서는 otx를 이용한 fine-tuning에 대해 학습이 부족하여 최종 프로젝트에서 적용해 볼 생각입니다. 
