from concurrent.futures import ThreadPoolExecutor
import cv2
import math
import glob
import os
import re
import threading

# human face.
human_cascade_f = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt2.xml')

# anime face.
anime_cascade_f = cv2.CascadeClassifier('haarcascades/lbpcascade_animeface.xml')

# find eyes.
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

# file_dirにオリジナルのサイズの画像を格納しているディレクトリのpath指定
# file_dir = 'html_data/images/'
file_dir = 'images/'

# resize_dirに出力先ディレクトリのpath指定
# resize_dir = 'html_data/resized_images/'
resize_dir = 'resized_images/'

# 定数定義
ESC_KEY = 27  # Escキー
INTERVAL = 1  # 待ち時間
FRAME_RATE = 20  # fps

WINDOW_NAME1 = "preview_!"
WINDOW_NAME2 = "face_view_!"
WINDOW_NAME3 = "frame_view_1"
WINDOW_NAME4 = "preview_2"
WINDOW_NAME5 = "face_view_2"
WINDOW_NAME6 = "frame_view_2"

def resize():
    files = os.listdir(file_dir)
    for i, file in enumerate(files):
        if i > 0:
            file_path = file_dir + '/' + file
            img = cv2.imread(file_path, cv2.IMREAD_COLOR)

            size = (28, 28)
            resize_img = cv2.resize(img, size)

            cv2.imwrite(resize_dir + '/' + file, resize_img)


def video_face(FILE_NAME, EXT, category, size):

    # ビデオファイル読み込み
    video = cv2.VideoCapture(FILE_NAME + EXT)
    end_flag, c_frame = video.read()

    # ウィンドウの準備
    cv2.namedWindow(WINDOW_NAME1)
    cv2.namedWindow(WINDOW_NAME2)
    cv2.namedWindow(WINDOW_NAME3)

    counter = 1

    # 変換処理ループ
    while end_flag:

        # フレーム表示
        cv2.imshow(WINDOW_NAME1, c_frame)

        # asyncio.ensure_future(video_face_check(c_frame, FILE_NAME, counter, category, size))
        video_face_check(c_frame, FILE_NAME, counter, category, size)
        counter += 1

        # Escキーで終了
        key = cv2.waitKey(INTERVAL)
        if key == ESC_KEY:
            break

        # 次のフレーム読み込み
        end_flag, c_frame = video.read()

    # 終了処理
    cv2.destroyAllWindows()
    video.release()


def video_face_check(frame=None, name="", frame_number=1, category="human", size=100):

    if frame is not None:

        if frame_number % FRAME_RATE == 0:

            cv2.imshow(WINDOW_NAME3, frame)

            if category == "human":
                faces = human_cascade_f.detectMultiScale(frame, 1.3, 5, minSize=(size, size))

            elif category == "anime":
                faces = anime_cascade_f.detectMultiScale(frame, 1.3, 5, minSize=(size, size))

            layer = 1

            for (x, y, w, h) in faces:

                try:
                    face_only = frame[y:y + h, x:x + w]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 0), 2)

                    cv2.imshow(WINDOW_NAME2, frame)

                    cv2.imwrite('%s%s_%s_%d_face.jpg' % (resize_dir, name, layer, frame_number), face_only)
                    print('%s%s_%s_%d_face.jpg' % (resize_dir, name, layer, frame_number))
                    layer += 1

                except Exception as e:
                    print(e)
                    continue
        else:
            pass

    else:
        pass


def face_check():

    for path_in in [x for x in glob.glob(file_dir + '*/*')]:

        try:

            image_name = re.search('.*/(?P<name>.*)', path_in).group('name')

            # 画像を読み込む
            img = cv2.imread(path_in)

            file_name, ext = os.path.splitext(os.path.basename(image_name))

            if not os.path.isdir(resize_dir):
                os.makedirs(resize_dir)
            else:
                pass
                # shutil.rmtree(folder_name)
                # os.makedirs(folder_name)

            rows, cols, colors = img.shape

            # 元画像の斜辺サイズの枠を作る(0で初期化)
            hypot = int(math.hypot(rows, cols))

            human_flag, anime_flag = False, False

            # 各loopで違う角度の回転行列をかけた結果のものに対して検出を試みる
            for deg in range(-30, 50, 10):
                M = cv2.getRotationMatrix2D((hypot * 0.5, hypot * 0.5), -deg, 1.0)
                rotated = cv2.warpAffine(img, M, (hypot, hypot))

                human_faces = human_cascade_f.detectMultiScale(rotated, 1.3, 5, minSize=(120, 120))
                anime_faces = anime_cascade_f.detectMultiScale(rotated, 1.3, 5, minSize=(120, 120))

                counter = 1

                if not human_flag:
                    for (x, y, w, h) in human_faces:
                        # cv2.rectangle(rotated, (x, y), (x + w, y + h), (0, 0, 0), 2)
                        face_only = rotated[y:y + h, x:x + w]
                        cv2.imwrite('%s%s_%s_face_only_%s.jpg' % (resize_dir, file_name, deg, counter), face_only)
                        human_flag = True
                        print('human_face: %s%s_%s_face_only_%s.jpg' % (resize_dir, deg, file_name, counter))
                        # eyes = eye_cascade.detectMultiScale(face_only)
                        # for (ex, ey, ew, eh) in eyes:
                        #     # 検知した目を矩形で囲む
                        #     # cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                        #     # color_rotated = cv2.cvtColor(face_only, cv2.COLOR_GRAY2RGB)
                        #     # 認識結果の保存
                        #     # cv2.imwrite('%s/%s_deg_rotated.jpg' % (folder_name, deg), rotated)
                        #     cv2.imwrite('%s%s_%s_face_only_%s.jpg' % (resize_dir, file_name, deg, counter), face_only)
                        #     flag = True
                        #     print('%s%s_%s_face_only_%s.jpg' % (resize_dir, deg, file_name, counter))
                        #     break
                        # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
                        # cv2.imshow("img", rotated)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                        counter += 1

                counter = 1

                if not anime_flag:
                    for (x, y, w, h) in anime_faces:
                        # cv2.rectangle(rotated, (x, y), (x + w, y + h), (0, 0, 0), 2)
                        face_only = rotated[y:y + h, x:x + w]
                        cv2.imwrite('%s%s_%s_face_only_%s.jpg' % (resize_dir, file_name, deg, counter), face_only)
                        anime_flag = True
                        print('anime_face: %s%s_%s_face_only_%s.jpg' % (resize_dir, deg, file_name, counter))
                        # eyes = eye_cascade.detectMultiScale(face_only)
                        # for (ex, ey, ew, eh) in eyes:
                        #     # 検知した目を矩形で囲む
                        #     # cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                        #     # color_rotated = cv2.cvtColor(face_only, cv2.COLOR_GRAY2RGB)
                        #     # 認識結果の保存
                        #     # cv2.imwrite('%s/%s_deg_rotated.jpg' % (folder_name, deg), rotated)
                        #     cv2.imwrite('%s%s_%s_face_only_%s.jpg' % (resize_dir, file_name, deg, counter), face_only)
                        #     flag = True
                        #     print('%s%s_%s_face_only_%s.jpg' % (resize_dir, deg, file_name, counter))
                        #     break
                        # cv2.namedWindow("img", cv2.WINDOW_NORMAL)
                        # cv2.imshow("img", rotated)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                        counter += 1

                if human_flag and anime_flag:
                    break

        except Exception as e:
            continue

if __name__ == '__main__':

    # pool = ThreadPoolExecutor(2)
    # pool.submit(video_face, "Summer_Wars", ".mp4", "anime", 50, 1)
    # pool.submit(video_face, "Initial_D", ".mkv", "anime", 50, 2)
    #
    # th1 = threading.Thread(target=video_face, name='anime1', args=("Summer_Wars", ".mp4", "anime", 50, 1))
    # print("th1 start.")
    # th1.start()
    #
    # th2 = threading.Thread(target=video_face, name='anime2', args=("test", ".avi", "anime", 50, 2))
    # th2.start()
    # print("th2 start.")
    # th2.join()
    #
    # print("all finished.")
    # th1.start()
    # th2.start()



    video_face("Summer_Wars", ".mp4", "anime", 50)
    # video_face("Initial_D", ".mkv", "anime", 50, 2)
    # video_face("test", ".avi", "anime", 50)
    video_face("test", ".mp4", "anime", 50)
    # video_face("test.mp4", "anime")
    # video_face("test.mov", "anime")
    # video_face("IMG_5941.mp4", "human")
    # video_face("Perfume2", ".mp4", "human")
    video_face("SilentSiren_Live_2015", ".mp4", "human", 100)
    # video_face("SilentSiren_Live_2016", ".mp4", "anime")
    # asyncio.ensure_future(video_face("Summer_Wars", ".mp4", "anime", 50, 1))
    # asyncio.ensure_future(video_face("Initial_D", ".mkv", "anime", 50, 2))
    # face_check()

