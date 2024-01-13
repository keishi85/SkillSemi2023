using OpenCvSharp;
using System.Text.RegularExpressions;
using System.Drawing.Imaging;
using System.Drawing;
using System.Windows.Forms;
//using OpenCvSharp.Extensions;
using System;
using System.Collections.Generic;
using System.IO;
using OpenCvSharp.Aruco;
using static System.Windows.Forms.VisualStyles.VisualStyleElement;

namespace ArucoPointer
{
	public class Calibration
	{
        private Form1 form;
        private VideoCapture capture;
        private Mat frame;
        private Bitmap image;
        private System.Windows.Forms.Timer timer;
        private List<Mat> rvecs, tvecs; // 回転ベクトルと並進ベクトルを保存するリスト
        private Dictionary arucoDict;
        private DetectorParameters arucoParams;
        private int frameIndex = 0;
        private int MaxImages = 20;
        private bool isCameraRunning = false;
        private bool isOriginSet;

        private List<(Mat rotationMatrix, Mat translationVector)> calibrationData = new List<(Mat rotationMatrix, Mat translationVector)>();
        private int calibrationCount = 0;
        private bool isCalibrated = false;
        private Vec3d markerToTipOffset; // 実測値に設定する必要あり
        private Mat cameraMatrix;
        private Mat distCoeffs;
        private Vec3d originPosition; // 原点位置
        private bool isTrackingActive = true;
        private System.Windows.Forms.Timer updateTimer;
        private Mat tipToMarkerTransform; // 変換行列を保存する

        public Calibration(Form1 form1)
		{
            this.form = form1;
            CalibrationProgram();
        }

        public void CalibrationProgram()
        {
            capture = new VideoCapture(0);
            frame = new Mat();
            form.pictureBox1.SizeMode = System.Windows.Forms.PictureBoxSizeMode.CenterImage;
            form.pictureBox2.SizeMode = System.Windows.Forms.PictureBoxSizeMode.CenterImage;

            // timer の初期化
            timer = new System.Windows.Forms.Timer();
            timer.Interval = 33;
            timer.Tick += new EventHandler(timer_Tick);
            timer.Start();
            arucoDict = CvAruco.GetPredefinedDictionary(PredefinedDictionaryName.Dict4X4_50);  // Dict4*4_50 : Aruco markerの種類
            rvecs = new List<Mat>();
            tvecs = new List<Mat>();
            Mat averageRvec = new Mat();
            Mat averageTvec = new Mat();

            markerToTipOffset = new Vec3d(0.0125, 0.142, 0); // 実測値/m
            DetectorParameters arucoParams = new DetectorParameters();
            // プログレスバーの初期化
            form.progressBar1.Maximum = MaxImages;
            form.progressBar1.Step = 1;
            form.progressBar2.Maximum = 10;

            updateTimer = new System.Windows.Forms.Timer();
            updateTimer.Interval = 100; // インターバルを100ミリ秒に設定
            updateTimer.Tick += new EventHandler(UpdateTimer_Tick); 
        }


        public void PerformCalibration()
        {
            capture.Open(0); // 0は通常デフォルトのカメラ 
            capture.Set(VideoCaptureProperties.FrameWidth, 1920); // 幅を設定 
            capture.Set(VideoCaptureProperties.FrameHeight, 1080); // 高さを設定
        }

        private void timer_Tick(object sender, EventArgs e)
        {
            if (!isCalibrated)
            {
                if (capture.Read(frame) && !frame.Empty())
                {
                    image = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(frame);
                    form.Invoke(new Action(() =>
                    {
                        form.pictureBox1.Image = image; // 画像を表示
                    }));
                }
            }
            else
            {
                if (capture.Read(frame) && !frame.Empty())
                {
                    image = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(frame);
                    form.Invoke(new Action(() =>
                    {
                       // form.pictureBox2.Image = image; // 画像を表示
                    }));
                }
            }

            
        }

        public void Cleanup()
        {
            timer.Stop(); // タイマーを停止
            capture.Release(); // カメラのリソースを解放
            frame.Dispose(); // Matのリソースを解放
            if (image != null)
                image.Dispose(); // イメージのリソースを解放
        }

        private void CalibrateCamera()
        {
            
            OpenCvSharp.Size patternSize = new OpenCvSharp.Size(9, 6);
            float squareSize = 10.0f;
            List<Mat> objectPoints = new List<Mat>();  
            List<Mat> imagePoints = new List<Mat>(); 

            // オブジェクトポイントの準備 
            Mat objp = new Mat(patternSize.Height * patternSize.Width, 1, MatType.CV_32FC3);
            for (int i = 0; i < patternSize.Height; i++)
            {
                for (int j = 0; j < patternSize.Width; j++)
                {
                    objp.Set(i * patternSize.Width + j, new Point3f(j * squareSize, i * squareSize, 0));
                }
            }

            // すべてのチェスボード画像ファイルのリストを取得 
            string[] imageFiles = Directory.GetFiles(@"./calibration", "*.png");

            foreach (var fileName in imageFiles)
            {
                using (var img = new Mat(fileName))
                {
                    if (img.Empty()) continue;
                    Point2f[] corners;
                    bool found = Cv2.FindChessboardCorners(img, patternSize, out corners);

                    if (found)
                    {
                        using (var grayImg = new Mat())
                        {
                            Cv2.CvtColor(img, grayImg, ColorConversionCodes.BGR2GRAY);
                            Cv2.CornerSubPix(grayImg, corners, new OpenCvSharp.Size(11, 11), new OpenCvSharp.Size(-1, -1),
                            new TermCriteria(CriteriaTypes.Eps | CriteriaTypes.MaxIter, 30, 0.1));

                            // 対応するオブジェクトポイントとイメージポイントを追加 
                            objectPoints.Add(objp);
                            imagePoints.Add(new Mat(corners.Length, 1, MatType.CV_32FC2, corners));
                        }
                    }
                }
            }

            // カメラパラメータを計算 
            OpenCvSharp.Size imageSize = new OpenCvSharp.Size(1920, 1080);
            Mat cameraMatrix = new Mat();
            Mat distCoeffs = new Mat();
            Mat[] rvecs, tvecs;
            double error = Cv2.CalibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix, distCoeffs, out rvecs, out tvecs);

            // カメラパラメータをファイルに保存 
            File.WriteAllText("./calibration/camera_matrix.txt", cameraMatrix.Dump());
            File.WriteAllText("./calibration/distortion_coeffs.txt", distCoeffs.Dump());
            }

        public void StartCalibration()
        {
            if (image != null)
            {
                // "calibration"フォルダーのパスを作成
                string folderPath = Path.Combine(Application.StartupPath, "calibration");
                if (!Directory.Exists(folderPath))
                {
                    // フォルダーが存在しない場合は作成する
                    Directory.CreateDirectory(folderPath);
                }

                // 画像ファイルの完全なパスを構築
                string fileName = $"Calibration_{frameIndex}.png";
                string filePath = Path.Combine(folderPath, fileName);

                // 画像を保存
                image.Save(filePath, ImageFormat.Png);

                // ファイルパスをlistBoxに追加
                form.listBox1.Items.Add(filePath);

                // 画像インデックスを増やす
                frameIndex++;

                // プログレスバーを更新
                if (form.progressBar1.Value < MaxImages)
                {
                    form.progressBar1.PerformStep();
                }

                // 最大画像数に達したかどうかをチェック
                if (form.progressBar1.Value == MaxImages)
                {
                    CalibrateCamera(); // キャリブレーションメソッドを呼び出す
                    MessageBox.Show("キャリブレーション完了！");
                    // プログレスバーと画像インデックスをリセット（必要なら）
                    form.progressBar1.Value = 0;
                    frameIndex = 0;
                }
            }
            else
            {
                MessageBox.Show("保存できる画像がありません！");
            }
        }

        private Mat ReadCameraMatrix(string filePath)
        {
            if (!File.Exists(filePath))
            {
                MessageBox.Show("ファイルが見つかりません: " + filePath);
                return new Mat();
            }

            string[] lines = File.ReadAllLines(filePath);
            if (lines.Length >= 1)
            {
                // すべての行のデータを1つの文字列に結合
                string combinedDataStr = string.Join(" ", lines)
                    .Trim(new char[] { '[', ']' })
                    .Replace(";", "")
                    .Replace(",", " ");
                double[] data = Array.ConvertAll(combinedDataStr.Split(new char[] { ' ' }, StringSplitOptions.RemoveEmptyEntries), double.Parse);

                // 行列のサイズを決定
                int rows = (filePath.Contains("camera_matrix")) ? 3 : 1;
                int cols = data.Length / rows;

                Mat matrix = new Mat(rows, cols, MatType.CV_64FC1, data);
                return matrix;
            }
            else
            {
                MessageBox.Show("ファイルの内容は行列を構築するのに十分ではありません。");
                return new Mat();
            }
        }

        private void StartDetectArrow()
        {
            // カメラパラメータを読み取る
            //cameraMatrix = ReadCameraMatrix("./calibration/camera_matrix.txt");
            //distCoeffs = ReadCameraMatrix("./calibration/distortion_coeffs.txt");

            cameraMatrix = new Mat(3, 3, MatType.CV_64FC1, new double[]
            {
                1.74355459e+03, 0.00000000e+00, 1.00776452e+03,
                0.00000000e+00, 1.72712508e+03, 5.26855534e+02,
                0.00000000e+00, 0.00000000e+00, 1.00000000e+00
             });

            distCoeffs = new Mat(1, 5, MatType.CV_64FC1, new double[]
            {
                -0.72920945, -0.13478921, 0.01676658, -0.05413861, 0.72264186
            });


            DetectorParameters arucoParams = new DetectorParameters(); // 直接インスタンス化

            // captureImageはカメラからのキャプチャと仮定
            Mat captureImage = CaptureImageFromCamera();

            // Arucoマーカーを検出
            Point2f[][] corners;
            int[] ids;
            Point2f[][] rejectedImgPoints;
            CvAruco.DetectMarkers(captureImage, arucoDict, out corners, out ids, arucoParams, out rejectedImgPoints);


            // デバッグ情報：cornersの内容を確認
            if (corners.Length > 0)
            {
                Console.WriteLine(corners.Length + "個のマーカーのコーナーが検出されました。");
                for (int i = 0; i < corners.Length; i++)
                {
                    Console.WriteLine("マーカー " + i + ": " + corners[i].Length + " 個のコーナー。");
                }
            }

            // マーカーが検出された場合、姿勢情報を抽出
            // Aruco markerに関する回転，並進ベクトルのみを抽出
            if (ids.Length > 0)
            {
                using (var rvecs = new Mat())
                using (var tvecs = new Mat())
                {
                    // EstimatePoseSingleMarkersを呼び出す
                    CvAruco.EstimatePoseSingleMarkers(corners, 0.05f, cameraMatrix, distCoeffs, rvecs, tvecs);

                    // ４つ角に対して以下の処理を行う
                    for (int i = 0; i < ids.Length; i++)
                    {
                        Vec3d rotationVector = rvecs.Get<Vec3d>(i);
                        Vec3d translationVector = tvecs.Get<Vec3d>(i);

                        // Vec3dをMatに変換し、Rodriguesメソッドを使用して回転ベクトルを回転行列に変換
                        var rotationMatrix = new Mat();
                        Cv2.Rodrigues(rotationVector, rotationMatrix);

                        // 検出結果を保存
                        double[] translationArray = { translationVector.Item0, translationVector.Item1, translationVector.Item2 };
                        calibrationData.Add((rotationMatrix, new Mat(3, 1, MatType.CV_64FC1, translationArray)));
                    }

                    // キャリブレーションカウントとプログレスバーを更新
                    calibrationCount++;
                    form.progressBar2.Value = calibrationCount;

                    // 必要な数のキャリブレーションが完了したか確認
                    if (calibrationCount >= 10)
                    {
                        CalculateTransformation();
                        isCalibrated = true;
                        MessageBox.Show("キャリブレーション完了、リアルタイムトラッキングを開始できます。");
                    }
                }
            }
            else
            {
                MessageBox.Show("有効なArucoマーカーが検出されなかったか、マーカーデータが不完全です。");
            }
        }

        private void CalculateTransformation()
        {
            if (calibrationData.Count == 0)
            {
                MessageBox.Show("計算に十分なデータがありません。");
                return;
            }

            // R' と T' を計算するためのアキュムレータを初期化します
            Mat rAccumulator = Mat.Eye(3, 3, MatType.CV_64FC1);
            Mat tAccumulator = new Mat(3, 1, MatType.CV_64FC1, 0);

            // calibrationData : 回転行列を並進行列
            foreach (var (rotationMatrix, translationVector) in calibrationData)
            {
                // 方程式に従ってアキュムレーションを行います
                rAccumulator = rAccumulator * rotationMatrix.Inv();
                tAccumulator = tAccumulator + (-rAccumulator * translationVector);
            }

            // 最終的に先端の位置を計算します
            Mat tipPosition = rAccumulator.Inv() * tAccumulator;
            // tipPosition を器具の先端の座標として使用します
        }

        public void StartCalibration2()
        {
            // カメラパラメータを読み込む
            //cameraMatrix = ReadCameraMatrix("./calibration/camera_matrix.txt");
            //distCoeffs = ReadCameraMatrix("./calibration/distortion_coeffs.txt");

            cameraMatrix = new Mat(3, 3, MatType.CV_64FC1, new double[]
            {
                1.74355459e+03, 0.00000000e+00, 1.00776452e+03,
                0.00000000e+00, 1.72712508e+03, 5.26855534e+02,
                0.00000000e+00, 0.00000000e+00, 1.00000000e+00
             });

            distCoeffs = new Mat(1, 5, MatType.CV_64FC1, new double[]
            {
                -0.72920945, -0.13478921, 0.01676658, -0.05413861, 0.72264186
            });

            DetectorParameters arucoParams = new DetectorParameters(); // 直接インスタンス化

            // captureImage はカメラからキャプチャされた画像と仮定
            Mat captureImage = CaptureImageFromCamera();

            // Aruco マーカーを検出
            Point2f[][] corners;
            int[] ids;
            Point2f[][] rejectedImgPoints;
            CvAruco.DetectMarkers(captureImage, arucoDict, out corners, out ids, arucoParams, out rejectedImgPoints);

            // マーカーが検出された場合、姿勢情報を抽出
            if (ids.Length > 0)
            {
                using (var rvecs = new Mat())
                using (var tvecs = new Mat())
                {
                    // EstimatePoseSingleMarkers を呼び出す
                    CvAruco.EstimatePoseSingleMarkers(corners, 0.02f, cameraMatrix, distCoeffs, rvecs, tvecs);

                    for (int i = 0; i < ids.Length; i++)
                    {
                        Vec3d rotationVector = rvecs.Get<Vec3d>(i);
                        Vec3d translationVector = tvecs.Get<Vec3d>(i);

                        // Vec3d を Mat に変換し、Rodrigues メソッドを使用して回転ベクトルを回転行列に変換
                        var rotationMatrix = new Mat();
                        Cv2.Rodrigues(rotationVector, rotationMatrix);

                        // 検出結果を保存
                        double[] translationArray = { translationVector.Item0, translationVector.Item1, translationVector.Item2 };
                        calibrationData.Add((rotationMatrix, new Mat(3, 1, MatType.CV_64FC1, translationArray)));
                    }

                    // キャリブレーションカウントとプログレスバーを更新
                    calibrationCount++;
                    form.progressBar2.Value = calibrationCount;

                    // 十分な回数のキャリブレーションが完了したかチェック
                    if (calibrationCount >= 10)
                    {
                        // 先端とマーカー間の固定位置関係を計算
                        tipToMarkerTransform = CalculateTipToMarkerTransform(calibrationData);
                        isCalibrated = true;
          
                        MessageBox.Show("キャリブレーション完了、リアルタイム追跡を開始できます。");
                    }
                }
            }
            else
            {
                MessageBox.Show("有効な Aruco マーカーが検出されないか、データが不完全です。");
            }
        }

        private Mat CalculateTipToMarkerTransform(List<(Mat rotationMatrix, Mat translationVector)> calibrationData)
        {
            // 累積行列とベクトルを初期化
            Mat A = new Mat(3 * calibrationData.Count, 6, MatType.CV_64FC1, Scalar.All(0));
            Mat B = new Mat(3 * calibrationData.Count, 1, MatType.CV_64FC1, Scalar.All(0));

            for (int i = 0; i < calibrationData.Count; i++)
            {
                var (rotationMatrix, translationVector) = calibrationData[i];

                // 回転行列と移動ベクトルを抽出
                double[] r = new double[3];
                double[] t = new double[3];
                for (int j = 0; j < 3; j++)
                {
                    r[j] = rotationMatrix.At<double>(j, 0);
                    t[j] = translationVector.At<double>(j, 0);
                }

                // A 回転行列を構築
                A.Set(i * 3, r[0]); A.Set(i * 3, 3, -1);
                A.Set(i * 3 + 1, r[1]); A.Set(i * 3 + 1, 4, -1);
                A.Set(i * 3 + 2, r[2]); A.Set(i * 3 + 2, 5, -1);

                // B 並進ベクトルを構築
                B.Set(i * 3, t[0] - markerToTipOffset.Item0);
                B.Set(i * 3 + 1, t[1] - markerToTipOffset.Item1);
                B.Set(i * 3 + 2, t[2] - markerToTipOffset.Item2);

            }
            
            // SVD を使用して最小二乗問題を解く
            // A * X = B を解く
            Mat X = new Mat();
            Cv2.Solve(A, B, X, DecompTypes.SVD);

            // 結果を抽出
            Mat transformationMatrix = Mat.Eye(4, 4, MatType.CV_64FC1);
            for (int i = 0; i < 3; i++)
            {
                transformationMatrix.At<double>(i, 3) = X.At<double>(i, 0);
            }

            return transformationMatrix;
        }

        public void StartMeasure()
        {
            if (!isCalibrated)
            {
                MessageBox.Show("まずキャリブレーションを完了してください。");
                return;
            }
            updateTimer.Start();
            isOriginSet = false; // 原点位置のフラグをリセット
        }

        // タイマーイベントの処理関数
        public void UpdateTimer_Tick(object sender, EventArgs e)
        {

            // リアルタイムで位置情報を表示
            Mat captureImage = CaptureImageFromCamera();
            //Mat captureImage = CaptureImageFromCamera();

            arucoParams = new DetectorParameters
            {
                MarkerBorderBits = 1 // 正の整数であることを保証
            };

            // Aruco マーカーを検出
            Point2f[][] corners;
            int[] ids;
            Point2f[][] rejectedImgPoints;
            CvAruco.DetectMarkers(captureImage, arucoDict, out corners, out ids, arucoParams, out rejectedImgPoints);

            if (ids.Length > 0)
            {
                // 回転ベクトルと移動ベクトルを抽出
                var rvecs = new Mat();
                var tvecs = new Mat();

                CvAruco.EstimatePoseSingleMarkers(corners, 0.02f, cameraMatrix, distCoeffs, rvecs, tvecs);
                // 座標軸を描画
                DrawAxis(captureImage, cameraMatrix, distCoeffs, rvecs, tvecs, ids);

                // 先端の位置を計算
                Vec3d worldTipPosition = CalculateRealTimeTipPosition(rvecs, tvecs, tipToMarkerTransform);

                // もし原点位置がまだ設定されていない場合、現在の先端位置を原点として設定
                if (!isOriginSet)
                {
                    originPosition = worldTipPosition; // 現在のフレームの世界座標系での先端位置を原点位置として設定
                    isOriginSet = true;
                }
                else
                {
                    double distance = CalculateDistance(originPosition, worldTipPosition); // 世界座標系での位置を使用して距離を計算
                    UpdateTipPositionDisplay(distance);
                }
            }
            else
            {
                updateTimer.Stop();
                MessageBox.Show("Aruco マーカーが検出されませんでした。");
            }

            // PictureBoxを更新して座標軸付きの画像を表示
            form.pictureBox2.Image = OpenCvSharp.Extensions.BitmapConverter.ToBitmap(captureImage);
            captureImage.Dispose();
        }

        private Vec3d CalculateRealTimeTipPosition(Mat rvecs, Mat tvecs, Mat tipToMarkerTransform)
        {
            Vec3d rvec = rvecs.Get<Vec3d>(0); // 最初のマーカーの回転ベクトル
            Vec3d tvec = tvecs.Get<Vec3d>(0); // 最初のマーカーの平行移動ベクトル

            // 回転ベクトルを回転行列に変換
            Mat rotationMatrix = new Mat();
            Cv2.Rodrigues(rvec, rotationMatrix);

            // 回転行列と平行移動ベクトルを組み合わせて4x4行列を作成
            Mat markerTransform = Mat.Eye(4, 4, MatType.CV_64FC1);
            rotationMatrix.CopyTo(markerTransform.SubMat(0, 3, 0, 3));
            new Mat(3, 1, MatType.CV_64FC1, new double[] { tvec.Item0, tvec.Item1, tvec.Item2 })
                .CopyTo(markerTransform.SubMat(0, 3, 3, 4));

            // tipToMarkerTransform行列を使用して、マーカー座標系内の先端位置を世界座標系に変換
            Mat worldTipPositionMat = markerTransform * tipToMarkerTransform;

            // 世界座標系内の先端位置を抽出
            Vec3d worldTipPosition = new Vec3d(
                worldTipPositionMat.At<double>(0, 3),
                worldTipPositionMat.At<double>(1, 3),
                worldTipPositionMat.At<double>(2, 3));

            return worldTipPosition;
        }

        private double CalculateDistance(Vec3d point1, Vec3d point2)
        {
            double dx = point1.Item0 - point2.Item0;
            double dy = point1.Item1 - point2.Item1;
            double dz = point1.Item2 - point2.Item2;
            return Math.Sqrt(dx * dx + dy * dy + dz * dz); // 2点間の距離を計算
        }

        private void UpdateTipPositionDisplay(double distance)
        {
            if (form.textBox1 != null && !form.textBox1.IsDisposed)
            {
                form.textBox1.Invoke((Action)(() => { form.textBox1.Text = $"距離: {distance * 100:F2} cm"; })); // 距離を表示
            }
        }


        private void DrawAxis(Mat image, Mat cameraMatrix, Mat distCoeffs, Mat rvecs, Mat tvecs, int[] ids)
        {
            float axisLength = 0.02f; // 座標軸の長さ

            for (int i = 0; i < ids.Length; i++)
            {
                // 3Dで座標軸のポイントを定義
                var axisPoints = new Point3f[]
                {
            new Point3f(0, 0, 0),  // 原点
            new Point3f(axisLength, 0, 0),  // X軸
            new Point3f(0, axisLength, 0),  // Y軸
            new Point3f(0, 0, axisLength)   // Z軸
                };

                // 3DポイントをMatオブジェクトに変換
                Mat objectPoints = new Mat(4, 1, MatType.CV_32FC3, axisPoints);

                // 画像平面に座標軸ポイントを投影
                Mat imagePointsMat = new Mat();
                Cv2.ProjectPoints(objectPoints, rvecs.Row(i), tvecs.Row(i), cameraMatrix, distCoeffs, imagePointsMat);

                // MatをPoint2f配列に変換
                var imagePoints = new Point2f[imagePointsMat.Rows];
                for (int j = 0; j < imagePointsMat.Rows; j++)
                {
                    var point = imagePointsMat.At<Vec2f>(j);
                    imagePoints[j] = new Point2f(point.Item0, point.Item1);
                }

                // 座標軸を描画
                Cv2.Line(image, new OpenCvSharp.Point(imagePoints[0].X, imagePoints[0].Y), new OpenCvSharp.Point(imagePoints[1].X, imagePoints[1].Y), Scalar.Red, 2); // X軸、赤色
                Cv2.Line(image, new OpenCvSharp.Point(imagePoints[0].X, imagePoints[0].Y), new OpenCvSharp.Point(imagePoints[2].X, imagePoints[2].Y), Scalar.Green, 2); // Y軸、緑色
                Cv2.Line(image, new OpenCvSharp.Point(imagePoints[0].X, imagePoints[0].Y), new OpenCvSharp.Point(imagePoints[3].X, imagePoints[3].Y), Scalar.Blue, 2);  // Z軸、青色
            }
        }



        public bool GetIsSave()
        {
            return isCalibrated;
        }

        public void StartCamera()
        {
            if (!isCameraRunning)
            {
                capture.Open(0); // カメラを起動
                if (capture.IsOpened())
                {
                    timer.Start(); // タイマーを開始
                    isCameraRunning = true;
                }
            }
        }

        public void StopCamera()
        {
            if (isCameraRunning)
            {
                timer.Stop(); // タイマーを停止
                capture.Release(); // カメラのリソースを解放
                isCameraRunning = false;

                // UIスレッドでpictureBox1の画像を消去
                form.Invoke(new Action(() =>
                {
                    form.pictureBox1.Image = null;
                }));
            }
        }


        private Mat CaptureImageFromCamera()
        {
            if (capture.IsOpened() && capture.Read(frame))
            {
                return frame.Clone();
            }
            return new Mat();
        }



        private void DetectAndComputeTransformation(Mat frame)
        {
            // Aruco markerの検出
            // corners : 画像上の各マーカーの角の位置, ids : 検出されたマーカーのID配列
           // CvAruco.DetectMarkers(frame, arucoDict, out Point2f[][] corners, out int[] ids);

           /* if (ids.Length > 0)
            {
                Mat rvec = new Mat();
                Mat tvec = new Mat();

                // Todo : 要変更
                // cameraMatrix, distCoeffsを定義
                double fx = frame.Width / 2; // 仮の焦点距離
                double fy = frame.Height / 2;
                double cx = frame.Width / 2; // 画像の中心
                double cy = frame.Height / 2;
                Mat cameraMatrix = new Mat(3, 3, MatType.CV_64FC1, new double[] { fx, 0, cx, 0, fy, cy, 0, 0, 1 });
                Mat distCoeffs = new Mat(1, 5, MatType.CV_64FC1, new double[5]);

                // 各マーカーに対して回転ベクトルと並進ベクトルを計算する
               // CvAruco.EstimatePoseSingleMarkers(corners, 0.05f, cameraMatrix, distCoeffs, rvec, tvec);

                // rvec と tvec をリストに保存
                rvecs.Add(rvec);
                tvecs.Add(tvec);
            }*

            /*
             * Todo : Markerが検出されなかった場合の処理
             */
        }

        // Todo : クォータニオン等を用いる方が精度向上が見込める
        /*
        private void CalculateAveragePose0()
        {
            // 平均の回転ベクトルと並進ベクトルを計算する
            foreach (var rvec in rvecs)
            {
                averageRvec += rvec;
            }
            averageRvec /= rvecs.Count;

            foreach (var tvec in tvecs)
            {
                averageTvec += tvec;
            }
            averageTvec /= tvecs.Count;

            // 平均の回転ベクトルを回転行列に変換する
            Mat rotationMatrix = new Mat();
            Cv2.Rodrigues(averageRvec, rotationMatrix);

            // 回転行列を使用したさらなる処理
            // ...
        }

        public Mat GetAverageRvec()
        {
            return averageRvec;
        }

        public Mat GetAverageTvec()
        {
            return averageTvec;
        }

        // Aruco marker から矢印の検出
        private void CalculateAveragePose()
        {
            Mat averageArrowPosition = new Mat(); // 矢印シールの平均位置

            for (int i = 0; i < rvecs.Count; i++)
            {
                Mat rvec = rvecs[i];
                Mat tvec = tvecs[i];

                // マーカーから矢印シールへの変換
                Mat arrowPosition = TransformMarkerToArrow(rvec, tvec, 0.10); // 10cmの距離
                averageArrowPosition += arrowPosition;
            }

            averageArrowPosition /= rvecs.Count;

            // 回転行列を計算する（必要に応じて）
            // ...
        }

        private Mat TransformMarkerToArrow(Mat rvec, Mat tvec, double distance)
        {
            // 回転ベクトルから回転行列への変換
            Cv2.Rodrigues(rvec, out Mat rotationMatrix);

            // マーカーから矢印までのオフセット（ローカル座標系）
            Mat arrowOffset = new Mat(3, 1, MatType.CV_64FC1, new double[] { 0, 0, -distance });

            // マーカーの向きに合わせてオフセットを回転させる
            Mat arrowPositionInLocal = rotationMatrix * arrowOffset;

            // マーカーの世界座標系における位置にオフセットを加算して矢印の位置を計算
            Mat arrowPositionInWorld = tvec + arrowPositionInLocal;

            return arrowPositionInWorld;
        }*/        
    }
} 