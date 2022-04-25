

</br>

![image](https://user-images.githubusercontent.com/82289435/161487725-cb433d95-1c59-47eb-b305-218a8c42ea46.png)

### 🚢 Members

[전다운](https://github.com/updaun)|[박정재](https://github.com/jeongjae96)|[김규민](https://github.com/km9mn)|[이융희](https://github.com/yoonghee)|[서경국](https://github.com/tjrudrnr2)|
:-:|:-:|:-:|:-:|:-:|
![image1][image1]|![image2][image2]|![image3][image3]|![image4][image4]|![image5][image5]|

[image1]: https://user-images.githubusercontent.com/82289435/161474965-fde57430-c7d8-4a8b-b042-a60e553cfb4e.png
[image2]: https://user-images.githubusercontent.com/82289435/161475112-33b5e095-c2f1-4ed8-90cb-c3ae9f6296ba.png
[image3]: https://user-images.githubusercontent.com/82289435/161475194-7b2f9f11-98fa-4c10-b3fa-ef986e8775d5.png
[image4]: https://user-images.githubusercontent.com/82289435/161475112-33b5e095-c2f1-4ed8-90cb-c3ae9f6296ba.png
[image5]: https://user-images.githubusercontent.com/82289435/161475256-bc796065-f8f8-4bdc-9d43-05b684a73d7d.png



### 🔥 Contribution  
- `전다운` &nbsp; Data Annotation • Git management • Data Annotation Output Training • Experiments combination 
- `박정재` &nbsp; Data Annotation • Git management • Polygon Processing • Data Annotation Output Training • Data Annotation Visualization 
- `김규민` &nbsp; Data Annotation • WandB Interlocking • Sweep • Data Storage • Polygon Processing • Data Annotation Output Training • Extra Data Convert Format
- `이융희` &nbsp; Data Annotation • Big Data Training • Shell Script Download • Convert MLT • Data Augmentation with BBox
- `서경국` &nbsp; Data Annotation • Git Slack Notification • Data Sorting Language • Early Stopping • Resume Training • Data Augmentation

</br>

### 🏆 LB Score

- Public LB (8등/19팀)
    - f1 : 0.7042, recall : 0.6199, precision : 0.8150
- Private LB (2등/19팀) 
    - f1 : 0.6966, recall : 0.6195, precision : 0.7956

![output](https://user-images.githubusercontent.com/82289435/164719211-6aaeb9a4-6c2b-4740-b844-de1dcd20ec69.png)

</br>

## 🔎 Text Detection for Optical Character Recognition 📑

</br>

![image](https://user-images.githubusercontent.com/82289435/164364294-f0065a31-52b1-46ca-8e65-af3874b247a6.png)

### 문제 정의 ❓

- 스마트폰으로 카드를 결제하거나, 카메라로 카드를 인식할 경우 자동으로 카드 번호가 입력되는 경우가 있습니다. 또 주차장에 들어가면 차량 번호가 자동으로 인식되는 경우도 흔히 있습니다.
- 이처럼 **OCR (Optimal Character Recognition)** 기술은 사람이 직접 쓰거나 이미지 속에 있는 문자를 얻은 다음 이를 컴퓨터가 인식할 수 있도록 하는 기술로, 컴퓨터 비전 분야에서 현재 널리 쓰이는 대표적인 기술 중 하나입니다.

</br>


### ⚙ Development Environment
- GPU : Nvidia Tesla V100
- OS : Linux Ubuntu 18.04
- Runtime : Python 3.8.5


</br>

### 💾 Dataset
![image](https://user-images.githubusercontent.com/82289435/164365009-99d65142-95e9-4cac-8f46-02072d26ecb0.png)

- Upstage Annotation Output Dataset(약 1300장)
- ICDAR 2017 multi-lingual(약 7000장)
- ICDAR 2019 multi-lingual(약 7000장)


### 📐 Metrics

- F1-Score

- DetEval

    - DetEval은, 이미지 레벨에서 정답 박스가 여러개 존재하고, 예측한 박스가 여러개가 있을 경우, 박스끼리의 다중 매칭을 허용하여 점수를 주는 평가방법 중 하나 입니다.

</br>

    1) 모든 정답/예측박스들에 대해서 Area Recall, Area Precision을 미리 계산해냅니다.

</br>
    - Area Recall = 정답과 예측박스가 겹치는 영역 / 정답 박스의 영역
</br>
    - Area Precision = 정답과 예측박스가 겹치는 영역 / 예측 박스의 영역

</br>

![image](https://user-images.githubusercontent.com/82289435/164357414-8b681c29-0026-44ae-9782-9987de160926.png)


</br>

    2) 모든 정답 박스와 예측 박스를 순회하면서, 매칭이 되었는지 판단하여 박스 레벨로 정답 여부를 측정합니다.

</br>



- 박스들이 매칭이 되는 조건
    - 박스들을 순회하면서, 위에서 계산한 Area Recall, Area Precision이 0 이상일 경우 매칭 여부를 판단
    - 박스의 정답 여부는 Area Recall 0.8 이상, Area Precision 0.4 이상을 기준으로 하고 있습니다.

- 매칭이 되었는가 대한 조건(3가지)

    - one-to-one match: 정답 박스 1개와 예측 박스 1개가 매칭 && 기본조건 성립

    - one-to-many match: 정답 박스 1개와 예측 박스 여러개가 매칭되는 경우

    - many-to-one match: 정답 박스 여러개와 예측박스 1개가 매칭되는 경우

    - one-to-many match 경우에 한해서, 박스 recall / precision 에 0.8 로 penalty가 적용됩니다.

![image](https://user-images.githubusercontent.com/82289435/164357662-e96c7125-c39c-4784-a931-9738fa46358d.png)


![image](https://user-images.githubusercontent.com/82289435/164358796-dda1bf36-11ff-45d9-af27-433207356cb3.png)


- G1과 P1은 one-to-one match가 성립되었고, Area Recall, Area Precision 모두 0.99로 threshold 이상이므로, 정답으로 책정

- G2, G3와 P2는 many-to-one match가 성립되었고, Area Recall 0.9(0.81+0.99)/2, Area Precision0.91(0.41+0.5) 로 threshold 이상으로 정답으로 책정

- G4와 P3, P4는 one-to-many match가 성립되었고, Area Recall 0.88(0.46+0.42), Area Precision 0.96(1.0+0.92)/2 로 threshold 이상으로 정답으로 책정



Recall, Precision, H-mean(F1 score)을 구해보면

- Recall = ( 1(G1) + 1(G2) + 1(G3) + 0.8(G4) ) / 4(len(gt)) = 0.95,

- Precision = ( 1(P1) + 1(P2) + 0.8(P3) + 0.8(P4) ) / 4(len(prediction)) = 0.9,

- H-mean = 2 * 0.95 * 0.9 / (0.95 + 0.9) = 0.92

해당 이미지에 대해서 최종적으로 0.92점을 부여

</br>

    3) 모든 이미지에 대하여 Recall, Precision을 구한 이후, 최종 F1-Score은 모든 이미지 레벨에서 측정 값의 평균으로 측정됩니다.

</br>

테스트 셋 모든 이미지들에 대해서 Recall, Precision로 점수를 구하고 평균내어 최종 점수를 구하게 됩니다.

예) image1, image2 두개의 테스트 이미지가 존재하고, 계산의 편의성을 위해서 image1은 위의 예시와 동일, image2는 정답/예측박스가 1개이고 맞았다고 가정하고 계산해보면

- Final Recall = 1 + 1 + 1 + 0.8 + 1 / 5 = 0.96

- Final Precision = 1 + 1 + 0.8 + 0.8 + 1 / 5 = 0.92

- Final F1 = 2 * 0.96 * 0.92 / (0.96 + 0.92) = 0.94

(분모가 5인 이유: image1에서 정답/예측박스 4개, image2에서 정답/예측박스가 1개이므로)

해당 테스트 셋에서 최종 점수는 0.94 점이 되겠습니다.

</br>



### Working Directory
```
.
|-- OCR_EDA.ipynb
|-- config.py
|-- convert_mlt.py
|-- dataset.py
|-- detect.py
|-- deteval.py
|-- east_dataset.py
|-- examples
|   |-- README.md
|   |-- EDA
|   |   |-- annotation_visualization.py
|   |   `-- readme.md
|   |-- data
|   |   |-- 04170019_Annotation_Review.csv
|   |   |-- readme.md
|   |   |-- revise_annotation.ipynb
|   |   `-- revised_train.json
|   |-- polygon
|   |   |-- README.md
|   |   |-- anti_empty.json
|   |   `-- dataset.py
|   |-- textinthewild
|   |   |-- README.md
|   |   `-- titw2ufo.ipynb
|   `-- wandb_train
|       |-- README.md
|       |-- config.py
|       `-- wandb_train.py
|-- inference.py
|-- loss.py
|-- mk_ICDAR
|   |-- convert_mlt_mixpass17.py
|   |-- convert_mlt_mixpass19.py
|   |-- download17.sh
|   |-- download19.sh
|   |-- readme.md
|   |-- urls17.txt
|   `-- urls19.txt
|-- model.py
|-- readme.md
|-- requirements.txt
`-- train.py
```

### 📢 Presentation
[하이파이프_발표자료.pdf](https://drive.google.com/file/d/1bf-Mzx0vUxqSR30JWG0zEUtXYyvrnToY/view?usp=sharing)

