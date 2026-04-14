import cv2 as cv
import numpy as np
import trimesh

# ── 체스보드 설정 ─────────────────────────────────────────────────
BOARD_PATTERN  = (7, 5)
BOARD_CELLSIZE = 0.025
BOARD_CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

obj_points = BOARD_CELLSIZE * np.array(
    [[c, r, 0] for r in range(BOARD_PATTERN[1]) for c in range(BOARD_PATTERN[0])],
    dtype=np.float32)


def calibrate_camera(video_file):
    """영상에서 자동으로 캘리브레이션"""
    print('[캘리브레이션 시작]')
    video = cv.VideoCapture(video_file)
    assert video.isOpened()

    img_points = []
    obj_pts_list = []
    total = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    frame_idx = 0

    while len(img_points) < 20:
        valid, img = video.read()
        if not valid:
            break
        frame_idx += 1
        if frame_idx % 10 != 0:  # 10프레임마다 하나씩
            continue

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        complete, corners = cv.findChessboardCorners(gray, BOARD_PATTERN, cv.CALIB_CB_FAST_CHECK)
        if complete:
            corners = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), BOARD_CRITERIA)
            img_points.append(corners)
            obj_pts_list.append(obj_points)
            print(f'  프레임 {frame_idx}/{total} 선택 (총 {len(img_points)}장)')

    video.release()
    assert len(img_points) >= 5, '캘리브레이션에 충분한 프레임이 없습니다!'

    img_size = gray.shape[::-1]
    rms, K, dist_coeff, _, _ = cv.calibrateCamera(obj_pts_list, img_points, img_size, None, None)

    print(f'\n## 캘리브레이션 결과')
    print(f'* RMS error = {rms:.4f}')
    print(f'* fx={K[0,0]:.2f}, fy={K[1,1]:.2f}, cx={K[0,2]:.2f}, cy={K[1,2]:.2f}')
    print(f'* dist = {dist_coeff.flatten()}')
    return K, dist_coeff


def load_glb(glb_path):
    scene = trimesh.load(glb_path)
    geo = list(scene.geometry.values())[0]

    # UV 텍스처에서 색상 샘플링
    tex_img = geo.visual.material.baseColorTexture
    tex_arr = np.array(tex_img)
    th, tw = tex_arr.shape[:2]
    uv = geo.visual.uv
    px = np.clip((uv[:, 0] * tw).astype(int), 0, tw - 1)
    py = np.clip(((1 - uv[:, 1]) * th).astype(int), 0, th - 1)
    colors = tex_arr[py, px, :3].astype(np.uint8)

    mesh = trimesh.util.concatenate(list(scene.geometry.values()))

    # -90도 회전 (세우기)
    rot = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
    mesh.apply_transform(rot)

    # 크기 조정
    scale = BOARD_CELLSIZE * 1.5 / max(mesh.bounding_box.extents)
    mesh.apply_scale(scale)

    # 중심 맞추기 (전체 축 기준)
    c = mesh.bounding_box.centroid
    mesh.apply_translation([-c[0], -c[1], -c[2]])
    # 발바닥(최대 z)을 보드 표면(z=0)에 맞춤 → 머리는 z<0 (카메라 방향)
    mesh.apply_translation([0, 0, -mesh.bounding_box.bounds[1][2]])

    # 체스보드 중앙으로
    cx = BOARD_CELLSIZE * (BOARD_PATTERN[0] - 1) / 2
    cy = BOARD_CELLSIZE * (BOARD_PATTERN[1] - 1) / 2
    mesh.apply_translation([cx, cy, 0])

    return np.array(mesh.vertices, dtype=np.float32), np.array(mesh.faces, dtype=np.int32), colors


def draw_model_filled(img, vertices, faces, colors, rvec, tvec, K, dist_coeff):
    pts_2d, _ = cv.projectPoints(vertices, rvec, tvec, K, dist_coeff)
    pts_2d = pts_2d.reshape(-1, 2)
    h, w = img.shape[:2]

    R, _ = cv.Rodrigues(rvec)
    verts_cam = (R @ vertices.T + tvec).T
    face_depths = verts_cam[faces, 2].mean(axis=1)
    sorted_idx = np.argsort(-face_depths)

    for idx in sorted_idx:
        face = faces[idx]
        pts = pts_2d[face].astype(np.int32)
        if np.any(pts[:, 0] < -200) or np.any(pts[:, 0] > w + 200):
            continue
        if np.any(pts[:, 1] < -200) or np.any(pts[:, 1] > h + 200):
            continue
        fc = colors[face].mean(axis=0).astype(int)
        bgr = (int(fc[2]), int(fc[1]), int(fc[0]))
        cv.fillPoly(img, [pts], bgr)


def draw_axes(img, rvec, tvec, K, dist_coeff):
    length = BOARD_CELLSIZE * 2
    axis_pts = np.float32([[0,0,0],[length,0,0],[0,length,0],[0,0,-length]])
    pts, _ = cv.projectPoints(axis_pts, rvec, tvec, K, dist_coeff)
    pts = pts.reshape(-1, 2).astype(np.int32)
    o = tuple(pts[0])
    cv.arrowedLine(img, o, tuple(pts[1]), (0,0,255), 2, cv.LINE_AA, tipLength=0.2)
    cv.arrowedLine(img, o, tuple(pts[2]), (0,255,0), 2, cv.LINE_AA, tipLength=0.2)
    cv.arrowedLine(img, o, tuple(pts[3]), (255,0,0), 2, cv.LINE_AA, tipLength=0.2)


if __name__ == '__main__':
    VIDEO_FILE = 'chessboard.mp4'
    GLB_FILE   = 'cat_plushie.glb'

    # 1) 캘리브레이션
    K, dist_coeff = calibrate_camera(VIDEO_FILE)

    # 2) GLB 모델 로드
    print('\nGLB 모델 로딩 중...')
    vertices, faces, colors = load_glb(GLB_FILE)
    print(f'  정점: {len(vertices)}, 면: {len(faces)}')

    # 3) AR 실행
    print('\n[AR 시작] ESC 종료')
    video = cv.VideoCapture(VIDEO_FILE)
    assert video.isOpened()

    fps    = video.get(cv.CAP_PROP_FPS)
    width  = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
    delay  = max(1, int(1000 / fps))

    OUTPUT_FILE = 'ar_output.mp4'
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    writer = cv.VideoWriter(OUTPUT_FILE, fourcc, fps, (width, height))
    print(f'  출력 파일: {OUTPUT_FILE}  ({width}x{height} @ {fps:.1f}fps)')

    prev_rvec = None
    prev_tvec = None

    while True:
        valid, img = video.read()
        if not valid:
            break

        complete, img_pts = cv.findChessboardCorners(img, BOARD_PATTERN, cv.CALIB_CB_FAST_CHECK)

        if complete:
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            img_pts = cv.cornerSubPix(gray, img_pts, (11,11), (-1,-1), BOARD_CRITERIA)

            # 코너 순서 정규화: 항상 img_pts[0]이 왼쪽에 오도록 강제
            # findChessboardCorners는 180° 뒤집힌 순서로 반환할 수 있어서
            # 좌표계 자체가 뒤집혀 고양이 방향이 바뀌는 문제를 방지
            if img_pts[0][0][0] > img_pts[-1][0][0]:
                img_pts = img_pts[::-1]

            if prev_rvec is not None:
                ret, rvec, tvec = cv.solvePnP(obj_points, img_pts, K, dist_coeff,
                                               rvec=prev_rvec.copy(), tvec=prev_tvec.copy(),
                                               useExtrinsicGuess=True)
            else:
                ret, rvec, tvec = cv.solvePnP(obj_points, img_pts, K, dist_coeff)

            if ret:
                prev_rvec = rvec.copy()
                prev_tvec = tvec.copy()
                draw_model_filled(img, vertices, faces, colors, rvec, tvec, K, dist_coeff)
                draw_axes(img, rvec, tvec, K, dist_coeff)

        writer.write(img)
        cv.imshow('AR - Cat Plushie', img)
        if cv.waitKey(delay) & 0xFF == 27:
            break

    writer.release()
    video.release()
    cv.destroyAllWindows()
    print(f'\n저장 완료: {OUTPUT_FILE}')
