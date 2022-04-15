### annotation_visualization.py
Annotation 시각화 코드입니다.

간단한 설명
- Illegibility가 true이면 box가 분홍색, false이면 하늘색으로 표시됩니다.
- 이미지 드래그 시, zoom-in 됩니다. 더블클릭하면 복구됩니다.

```
--root_dir: 이미지 경로
--annotation_file_name: annotation 파일명 (.json 생략)
--image_h: 출력 이미지의 세로 크기 (종횡비 유지)
```

Reference
- _https://stages.ai/competitions/184/discussion/talk/post/1274_
