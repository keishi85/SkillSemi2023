namespace ArucoPointer;
using System;
using System.Windows.Forms;


public partial class Form1 : Form
{
    // Calibration.csの呼び出し
    private Calibration calibrate;

    public Form1()
    {
        InitializeComponent();
        calibrate = new Calibration(this);

        // Load イベントハンドラを追加
        this.Load += new EventHandler(Form1_Load);
    }

    private void Form1_Load(object sender, EventArgs e)
    {
        // アプリケーション起動時にカメラからの画像取得と表示を開始
        calibrate.PerformCalibration();
        //progressBar1.Value = 0; // ProgressBar の値をリセット
    }

    private void bindingSource1_CurrentChanged(object sender, EventArgs e)
    {

    }

    private void textBox1_TextChanged(object sender, EventArgs e)
    {

    }

    private void textBox1_TextChanged_1(object sender, EventArgs e)
    {

    }

    private void labelCalibration_Click(object sender, EventArgs e)
    {

    }
    private void button1_Click(object sender, EventArgs e)
    {
    }

    private void button2_Click(object sender, EventArgs e)
    {
        // pivot運動から行列の算出
       // progressBar1.Visible = false;
        //progressBar2.Value = 0;
        progressBar2.Visible = true;
        calibrate.StartCalibration2();
    }

    public void textBox1_TextChanged_2(object sender, EventArgs e)
    {

    }

    private void progressBar1_Click(object sender, EventArgs e)
    {

    }

    private void listBox1_SelectedIndexChanged(object sender, EventArgs e)
    {

    }

    private void キャリブレーション_Click(object sender, EventArgs e)
    {

    }

    private void button4_Click(object sender, EventArgs e)
    {
        // Calibrationのやり直し
        progressBar2.Value = 0; // ProgressBar の値をリセット
        calibrate.PerformCalibration();
        calibrate.StartCalibration();
    }

    private void textBox1_TextChanged_3(object sender, EventArgs e)
    {

    }

    private void button4_Click_1(object sender, EventArgs e)
    {
        calibrate.StartCamera();
    }

    private void button5_Click(object sender, EventArgs e)
    {
        calibrate.StopCamera();
    }

    private void button6_Click(object sender, EventArgs e)
    {
       // progressBar1.Value = 0; // ProgressBar の値をリセット
       // progressBar1.Visible = false; // ProgressBar を表示
        button6.Visible = false;
        button2.Visible = true;
    }

    private void pictureBox1_Click(object sender, EventArgs e)
    {

    }

    private void button6_Click_1(object sender, EventArgs e)
    // カメラの内部パラメータの算出
    {
        // 進行状況バーの初期化や処理の開始
        //progressBar1.Visible = true; // ProgressBar を表示
        calibrate.PerformCalibration();
        calibrate.StartCalibration();
    }

    private void progressBar2_Click(object sender, EventArgs e)
    {

    }

    private void progressBar2_Click_1(object sender, EventArgs e)
    {

    }

    private void button8_Click(object sender, EventArgs e)
    {
        // 計測スタート
        calibrate.StartMeasure();
    }

    private void textBox1_TextChanged_4(object sender, EventArgs e)
    {

    }

    private void button7_Click(object sender, EventArgs e)
    {
        // 基準点の設定
        calibrate.UpdateOriginalPoint();
    }

    private void pictureBox2_Click(object sender, EventArgs e)
    {

    }

    private void label1_Click(object sender, EventArgs e)
    {
    }

    private void button6_Click_2(object sender, EventArgs e)
    {
        // Calibrationのやり直し
        calibrate.ResetCalibration();
        progressBar2.Value = 0; // ProgressBar の値をリセット
        //calibrate.PerformCalibration();
        //calibrate.StartCalibration();
        button6.Visible = false;
        button2.Visible = true;
    }
}