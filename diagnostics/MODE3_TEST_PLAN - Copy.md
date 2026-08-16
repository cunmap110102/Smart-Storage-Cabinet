# Mode 3 Test Plan

## 1. Muc tieu

Tai lieu nay dung de danh gia `mode 3` cua du an theo 3 nhom:

- Do on dinh detect mat
- Do chinh xac va do on dinh recognition
- Nguyen nhan gay bien dong ket qua va huong cai thien

Tai lieu nay bam theo code hien tai trong:

- `recognize_face.py`
- `mode3_detect_diagnostics.py`
- `mode3_recognition_diagnostics.py`
- `locker_ui.py`

## 2. Cac nguong hien tai trong code can theo doi

### 2.1. Detect va quality gate

- `MIN_FACE_BRIGHTNESS = 50.0`
- `MIN_FACE_SHARPNESS = 25.0`
- `MIN_FACE_WIDTH_RATIO = 0.16`
- `MAX_FACE_WIDTH_RATIO = 0.50`
- `SINGLE_FACE_STABLE_SECONDS = 0.35`
- `MULTI_FACE_BLOCK_SECONDS = 1.0`
- `TRACK_REFRESH_SECONDS = 1.0`
- `TRACK_TEMPLATE_MATCH_THRESHOLD = 0.45`
- `MAX_FACE_CENTER_SHIFT_RATIO = 0.10`
- `FRAME_PROCESS_SCALE = 0.5`

### 2.2. Recognition va mo tu

- Threshold embedding mac dinh: `0.42`
- Take mode:
- `TAKE_CONFIDENCE_THRESHOLD = 68.0`
- `TAKE_MASKED_CONFIDENCE_THRESHOLD = 70.0`
- `TAKE_REVIEW_CONFIDENCE_THRESHOLD = 58.0`
- `TAKE_REVIEW_MASKED_CONFIDENCE_THRESHOLD = 60.0`
- `TAKE_RECHECK_ACCEPT_CONFIDENCE_THRESHOLD = 65.0`
- `TAKE_RECHECK_ACCEPT_MASKED_CONFIDENCE_THRESHOLD = 61.0`
- `RECOGNITION_TIMEOUT_SECONDS = 3`
- `TAKE_REVIEW_TIMEOUT_SECONDS = 2`

## 3. Chi so can do

### 3.1. Detect metrics

- `frame_count`
- `no_face_frames`
- `multi_face_frames`
- `single_face_frames`
- `aligned_frames`
- `misaligned_frames`
- `low_light_frames`
- `blur_frames`
- `too_far_frames`
- `too_close_frames`
- `accepted_normal_frames`
- `accessory_frames`
- `brightness_mean/min/max`
- `sharpness_mean/min/max`
- `face_width_ratio_mean/min/max`
- `detector_counts`
- `state_counts`

### 3.2. Recognition metrics

- `trial_index`
- `condition`
- `expected_name`
- `dominant_state`
- `final_name`
- `final_confidence`
- `distance_mean/min/max`
- `recognition_attempts`
- `confirmed_match_frames`
- `mode3_decision`
- `expected_match`

### 3.3. Chi so tong hop nen tinh them sau khi test

- `Pass rate = pass_direct / total trials`
- `Review rate = review_zone / total trials`
- `Reject rate = unknown_or_rejected / total trials`
- `True Accept Rate`
- `False Reject Rate`
- `False Accept Rate`
- `Mean time-to-pass`
- `Condition-wise pass rate`

## 4. Quy tac chay test

- Giu nguyen camera, do phan giai, vi tri lap dat trong 1 dot test
- Moi dieu kien test nen chay it nhat `20 trials / nguoi`
- Nen co it nhat `10 nguoi` neu muon danh gia threshold mo tu
- Tat moi thay doi thu cong vao model trong luc benchmark
- Ghi ro thoi diem test: sang, trua, toi
- Ghi ro camera index, vi tri camera, khoang cach nguoi dung den camera

Luu y:

- De tranh ket qua bi "hoc them" trong luc benchmark, nen uu tien test tren bo du lieu on dinh va xem xet tam thoi vo hieu hoa online learning khi danh gia chinh thuc.
- Neu khong tat online learning, can ghi chu ro trial nao xay ra sau khi gallery da duoc cap nhat them.

## 5. Ma tran test de xuat

### 5.1. Detect stability test

| Nhom | Condition label | Mo ta | Thoi luong moi lan |
| --- | --- | --- | --- |
| Baseline | `baseline_front_good` | Anh sang tot, mat thang, 1 nguoi, dung khoang cach vua | 12s |
| Lighting | `low_light_front` | Anh toi | 12s |
| Lighting | `back_light_front` | Nguoc sang, nen sau sang manh | 12s |
| Lighting | `side_light_left` | Sang lech trai | 12s |
| Lighting | `side_light_right` | Sang lech phai | 12s |
| Distance | `too_far_borderline` | Dung sat nguong xa | 12s |
| Distance | `ideal_distance` | Vung toi uu | 12s |
| Distance | `too_close_borderline` | Dung sat nguong gan | 12s |
| Pose | `yaw_left_15` | Quay trai nhe | 12s |
| Pose | `yaw_right_15` | Quay phai nhe | 12s |
| Pose | `yaw_left_30` | Quay trai ro hon | 12s |
| Pose | `yaw_right_30` | Quay phai ro hon | 12s |
| Motion | `small_motion` | Lac dau nhe | 12s |
| Motion | `large_motion` | Di chuyen nhieu | 12s |
| Occlusion | `glasses_front` | Deo kinh | 12s |
| Occlusion | `mask_front` | Deo khau trang | 12s |
| Occlusion | `mask_glasses_front` | Deo ca khau trang va kinh | 12s |
| Crowd | `two_faces_frame` | Co 2 nguoi cung vao khung | 12s |

### 5.2. Recognition stability test

| Nhom | Condition label | Expected | Trials/goi y |
| --- | --- | --- | --- |
| Positive baseline | `rec_baseline_front_good` | Dung nguoi da dang ky | 20 |
| Positive lighting | `rec_low_light_front` | Dung nguoi da dang ky | 20 |
| Positive pose | `rec_yaw_15` | Dung nguoi da dang ky | 20 |
| Positive pose | `rec_yaw_30` | Dung nguoi da dang ky | 20 |
| Positive blur/motion | `rec_small_motion` | Dung nguoi da dang ky | 20 |
| Positive glasses | `rec_glasses_front` | Dung nguoi da dang ky | 20 |
| Positive mask | `rec_mask_front` | Dung nguoi da dang ky | 20 |
| Positive mask+glasses | `rec_mask_glasses_front` | Dung nguoi da dang ky | 20 |
| Negative impostor | `rec_impostor_clean` | Nguoi khac khong thuoc tu nay | 20 |
| Negative similar look | `rec_impostor_similar` | Nguoi khac co ngoai hinh gan giong | 20 |
| Negative crowd | `rec_two_faces_frame` | Co them nguoi phu trong khung | 20 |

## 6. Lenh chay de xuat

### 6.1. Detect diagnostics

```powershell
python mode3_detect_diagnostics.py --condition baseline_front_good --duration 12 --camera-index 0
python mode3_detect_diagnostics.py --condition low_light_front --duration 12 --camera-index 0
python mode3_detect_diagnostics.py --condition glasses_front --duration 12 --camera-index 0
python mode3_detect_diagnostics.py --condition two_faces_frame --duration 12 --camera-index 0
```

### 6.2. Recognition diagnostics

Thay `01` bang ID/locker duoc dang ky dung voi nguoi test.

```powershell
python mode3_recognition_diagnostics.py --expected-name 01 --condition rec_baseline_front_good --trials 20 --timeout 3 --min-display 3 --threshold 0.42 --camera-index 0
python mode3_recognition_diagnostics.py --expected-name 01 --condition rec_low_light_front --trials 20 --timeout 3 --min-display 3 --threshold 0.42 --camera-index 0
python mode3_recognition_diagnostics.py --expected-name 01 --condition rec_glasses_front --trials 20 --timeout 3 --min-display 3 --threshold 0.42 --camera-index 0
python mode3_recognition_diagnostics.py --expected-name 01 --condition rec_mask_front --trials 20 --timeout 3 --min-display 3 --threshold 0.42 --camera-index 0
```

### 6.3. Recognition test cho impostor

Van dung `--expected-name 01`, nhung nguoi dung truoc camera la nguoi khac.

```powershell
python mode3_recognition_diagnostics.py --expected-name 01 --condition rec_impostor_clean --trials 20 --timeout 3 --min-display 3 --threshold 0.42 --camera-index 0
```

## 7. Tieu chi danh gia pass/fail de xuat

### 7.1. Muc can dat cho detect baseline

- `single_face_frames / frame_count >= 0.85`
- `no_face_frames / frame_count <= 0.10`
- `misaligned_frames / single_face_frames <= 0.15`
- `low_light_frames = 0` trong baseline
- `blur_frames = 0` hoac rat thap trong baseline

### 7.2. Muc can dat cho recognition baseline

- `pass_direct rate >= 0.90`
- `expected_match = yes` dat it nhat `90%`
- `unknown_or_rejected <= 10%`
- `final_confidence mean >= 68`
- `distance_mean` phai on dinh, khong co nhieu trial dao dong sat threshold

### 7.3. Muc can dat cho impostor

- `final_name = expected_name` phai gan nhu bang `0`
- `pass_direct = 0`
- Neu co mo nham, can xem lai threshold truoc khi dua vao van hanh that

## 8. Cach doc ket qua

### 8.1. Neu detect tot nhung recognition kem

Nguyen nhan thuong gap:

- gallery chua du da dang
- threshold embedding chua phu hop
- online learning lam gallery lech
- state classifier danh nham `mask/glasses`
- anh tuy detect duoc nhung quality chua du cho matching

### 8.2. Neu detect khong on dinh

Nguyen nhan thuong gap:

- camera / anh sang thay doi manh
- mat nam sat bien nguong `brightness`, `sharpness`, `face_width_ratio`
- tracker drift
- detector `YuNet` co luc khong ra mat va code hien tai khong fallback tiep sang HOG khi `YuNet` da ton tai nhung tra rong

### 8.3. Dau hieu threshold dang dat chua dung

- Nhieu trial dung nguoi nhung `distance_mean` sat `0.42`
- `final_confidence` thuong xuyen nam trong vung `58-68`
- impostor co confidence khong thap hon nhieu so voi genuine

## 9. Khuyen nghi phan tich sau test

Sau moi dot test, nen tong hop thanh bang:

| Condition | Trials | Pass direct | Review zone | Reject | Mean conf | Mean distance | Ket luan |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rec_baseline_front_good | 20 |  |  |  |  |  |  |
| rec_low_light_front | 20 |  |  |  |  |  |  |
| rec_glasses_front | 20 |  |  |  |  |  |  |
| rec_mask_front | 20 |  |  |  |  |  |  |
| rec_impostor_clean | 20 |  |  |  |  |  |  |

Va bang detect:

| Condition | Frame count | No face | Multi face | Low light | Blur | Mean brightness | Mean sharpness | Ket luan |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_front_good |  |  |  |  |  |  |  |  |
| low_light_front |  |  |  |  |  |  |  |  |
| glasses_front |  |  |  |  |  |  |  |  |
| two_faces_frame |  |  |  |  |  |  |  |  |

## 10. Uu tien cai thien sau khi co so lieu

### Muc 1. Sua logic detector truoc

- Cho `YuNet` fallback sang `HOG` khi `YuNet` ton tai nhung khong tim thay mat
- Ghi log ly do reject theo frame
- Tam tat online learning khi benchmark

### Muc 2. Hieu chinh threshold bang du lieu that

- Ve phan bo genuine va impostor distances
- Chon lai threshold embedding thay vi giu co dinh `0.42`
- Chon lai `68/58/65` dua tren FAR va FRR mong muon

### Muc 3. Nang cap detector/alignment

- De xuat: `RetinaFace`
- Ly do: detect va landmark/alignment tot hon trong dieu kien pose, occlusion, anh sang phuc tap
- Bai bao: Deng et al., RetinaFace, CVPR 2020
- Link: https://openaccess.thecvf.com/content_CVPR_2020/html/Deng_RetinaFace_Single-Shot_Multi-Level_Face_Localisation_in_the_Wild_CVPR_2020_paper.html

### Muc 4. Nang cap embedding recognition

- De xuat: `ArcFace`
- Ly do: embedding discriminative tot hon so voi cach matching co ban dua tren khoang cach nhu hien tai
- Bai bao: Deng et al., ArcFace, CVPR 2019
- Link: https://openaccess.thecvf.com/content_CVPR_2019/html/Deng_ArcFace_Additive_Angular_Margin_Loss_for_Deep_Face_Recognition_CVPR_2019_paper.html

### Muc 5. Dua quality-aware vao quyet dinh

- De xuat: su dung score quality hoac feature-norm de dieu chinh nguong
- Bai bao:
- MagFace, CVPR 2021
- AdaFace, CVPR 2022
- CR-FIQA, CVPR 2023
- Links:
- https://openaccess.thecvf.com/content/CVPR2021/html/Meng_MagFace_A_Universal_Representation_for_Face_Recognition_and_Quality_Assessment_CVPR_2021_paper.html
- https://openaccess.thecvf.com/content/CVPR2022/html/Kim_AdaFace_Quality_Adaptive_Margin_for_Face_Recognition_CVPR_2022_paper.html
- https://openaccess.thecvf.com/content/CVPR2023/papers/Boutros_CR-FIQA_Face_Image_Quality_Assessment_by_Learning_Sample_Relative_Classifiability_CVPR_2023_paper.pdf

### Muc 6. Cai thien kha nang chiu occlusion

- De xuat: bo sung du lieu masked face that hoac synthetic, va xem xet embedding occlusion-aware
- Bai bao:
- Xu et al., OREO, CVPRW 2020
- Song et al., Occlusion Robust Face Recognition, ICCV 2019
- Huang et al., Masked Face Recognition Datasets and Validation, ICCVW 2021
- Links:
- https://openaccess.thecvf.com/content_CVPRW_2020/html/w48/Xu_On_Improving_the_Generalization_of_Face_Recognition_in_the_Presence_CVPRW_2020_paper.html
- https://openaccess.thecvf.com/content_ICCV_2019/html/Song_Occlusion_Robust_Face_Recognition_Based_on_Mask_Learning_With_Pairwise_ICCV_2019_paper.html
- https://openaccess.thecvf.com/content/ICCV2021W/MFR/html/Huang_Masked_Face_Recognition_Datasets_and_Validation_ICCVW_2021_paper.html

## 11. Thu tu thuc hien goi y

1. Chay detect baseline va low-light de xem gate quality co dang qua chat khong.
2. Chay recognition baseline cho 1 nguoi da dang ky.
3. Chay impostor test de uoc luong rui ro mo nham.
4. Chay glasses va mask test de xac dinh state classifier va luong suy giam.
5. Tong hop CSV thanh bang so sanh theo condition.
6. Neu can, hieu chinh threshold truoc khi doi model.

## 12. Dau ra mong muon sau dot test dau tien

Sau dot test thu 1, can co it nhat:

- `mode3_detect_results.csv`
- `mode3_detect_summary.md`
- `mode3_recognition_results.csv`
- `mode3_recognition_summary.md`
- 1 bang tong hop pass/reject theo tung condition
- 1 ket luan ro rang: van de chinh nam o detect, quality gate, state classifier, hay threshold recognition
