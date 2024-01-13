namespace ArucoPointer
{
    partial class Form1
    {
        /// <summary>
        ///  Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;
        private Label labelCalibration;
        private Label labelMeasurement;

        /// <summary>
        ///  Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        ///  Required method for Designer support - do not modify
        ///  the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            components = new System.ComponentModel.Container();
            button1 = new Button();
            button3 = new Button();
            calibration = new TabControl();
            キャリブレーション = new TabPage();
            button6 = new Button();
            progressBar2 = new ProgressBar();
            pictureBox1 = new PictureBox();
            listBox1 = new ListBox();
            button2 = new Button();
            labelCalibration = new Label();
            button5 = new Button();
            button4 = new Button();
            計測 = new TabPage();
            button8 = new Button();
            label2 = new Label();
            textBox1 = new TextBox();
            label1 = new Label();
            button7 = new Button();
            pictureBox2 = new PictureBox();
            labelMeasurement = new Label();
            imageList1 = new ImageList(components);
            calibration.SuspendLayout();
            キャリブレーション.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)pictureBox1).BeginInit();
            計測.SuspendLayout();
            ((System.ComponentModel.ISupportInitialize)pictureBox2).BeginInit();
            SuspendLayout();
            // 
            // button1
            // 
            button1.Location = new Point(0, 0);
            button1.Margin = new Padding(3, 4, 3, 4);
            button1.Name = "button1";
            button1.Size = new Size(86, 31);
            button1.TabIndex = 5;
            // 
            // button3
            // 
            button3.Location = new Point(0, 0);
            button3.Margin = new Padding(3, 4, 3, 4);
            button3.Name = "button3";
            button3.Size = new Size(86, 31);
            button3.TabIndex = 4;
            // 
            // calibration
            // 
            calibration.Controls.Add(キャリブレーション);
            calibration.Controls.Add(計測);
            calibration.Location = new Point(0, 0);
            calibration.Margin = new Padding(3, 4, 3, 4);
            calibration.Name = "calibration";
            calibration.SelectedIndex = 0;
            calibration.Size = new Size(913, 597);
            calibration.TabIndex = 3;
            // 
            // キャリブレーション
            // 
            キャリブレーション.Controls.Add(button6);
            キャリブレーション.Controls.Add(progressBar2);
            キャリブレーション.Controls.Add(pictureBox1);
            キャリブレーション.Controls.Add(listBox1);
            キャリブレーション.Controls.Add(button2);
            キャリブレーション.Controls.Add(labelCalibration);
            キャリブレーション.Controls.Add(button5);
            キャリブレーション.Controls.Add(button4);
            キャリブレーション.Location = new Point(4, 29);
            キャリブレーション.Margin = new Padding(3, 4, 3, 4);
            キャリブレーション.Name = "キャリブレーション";
            キャリブレーション.Padding = new Padding(3, 4, 3, 4);
            キャリブレーション.Size = new Size(905, 564);
            キャリブレーション.TabIndex = 0;
            キャリブレーション.Text = "キャリブレーション";
            キャリブレーション.UseVisualStyleBackColor = true;
            キャリブレーション.Click += キャリブレーション_Click;
            // 
            // button6
            // 
            button6.Font = new Font("Yu Gothic UI", 15F, FontStyle.Regular, GraphicsUnit.Point);
            button6.Location = new Point(597, 205);
            button6.Name = "button6";
            button6.Size = new Size(277, 85);
            button6.TabIndex = 12;
            button6.Text = "再測定";
            button6.UseVisualStyleBackColor = true;
            button6.Visible = false;
            button6.Click += button6_Click_2;
            // 
            // progressBar2
            // 
            progressBar2.Location = new Point(597, 124);
            progressBar2.Name = "progressBar2";
            progressBar2.Size = new Size(277, 43);
            progressBar2.TabIndex = 11;
            progressBar2.Visible = false;
            progressBar2.Click += progressBar2_Click_1;
            // 
            // pictureBox1
            // 
            pictureBox1.Location = new Point(6, 80);
            pictureBox1.Name = "pictureBox1";
            pictureBox1.Size = new Size(533, 381);
            pictureBox1.TabIndex = 2;
            pictureBox1.TabStop = false;
            pictureBox1.Click += pictureBox1_Click;
            // 
            // listBox1
            // 
            listBox1.FormattingEnabled = true;
            listBox1.ItemHeight = 20;
            listBox1.Location = new Point(109, 205);
            listBox1.Name = "listBox1";
            listBox1.Size = new Size(276, 184);
            listBox1.TabIndex = 6;
            // 
            // button2
            // 
            button2.Font = new Font("Yu Gothic UI", 15F, FontStyle.Regular, GraphicsUnit.Point);
            button2.Location = new Point(597, 205);
            button2.Name = "button2";
            button2.Size = new Size(277, 85);
            button2.TabIndex = 1;
            button2.Text = "キャリブレーション";
            button2.UseVisualStyleBackColor = true;
            button2.Click += button2_Click;
            // 
            // labelCalibration
            // 
            labelCalibration.AutoSize = true;
            labelCalibration.Font = new Font("Yu Gothic UI", 17F, FontStyle.Regular, GraphicsUnit.Point);
            labelCalibration.Location = new Point(0, 16);
            labelCalibration.Name = "labelCalibration";
            labelCalibration.Size = new Size(927, 40);
            labelCalibration.TabIndex = 0;
            labelCalibration.Text = "キャリブレーション : 指示棒でpivot操作の準備ができたら押してください（10回）\r\n";
            labelCalibration.Click += labelCalibration_Click;
            // 
            // button5
            // 
            button5.Location = new Point(736, 124);
            button5.Name = "button5";
            button5.Size = new Size(138, 43);
            button5.TabIndex = 9;
            button5.Text = "カメラオフ";
            button5.UseVisualStyleBackColor = true;
            button5.Click += button5_Click;
            // 
            // button4
            // 
            button4.Location = new Point(597, 124);
            button4.Name = "button4";
            button4.Size = new Size(138, 43);
            button4.TabIndex = 8;
            button4.Text = "カメラオン";
            button4.UseVisualStyleBackColor = true;
            button4.Click += button4_Click_1;
            // 
            // 計測
            // 
            計測.Controls.Add(button8);
            計測.Controls.Add(label2);
            計測.Controls.Add(textBox1);
            計測.Controls.Add(label1);
            計測.Controls.Add(button7);
            計測.Controls.Add(pictureBox2);
            計測.Controls.Add(labelMeasurement);
            計測.Font = new Font("Yu Gothic UI", 20F, FontStyle.Regular, GraphicsUnit.Point);
            計測.Location = new Point(4, 29);
            計測.Margin = new Padding(3, 4, 3, 4);
            計測.Name = "計測";
            計測.Padding = new Padding(3, 4, 3, 4);
            計測.Size = new Size(905, 564);
            計測.TabIndex = 1;
            計測.Text = "計測";
            計測.UseVisualStyleBackColor = true;
            // 
            // button8
            // 
            button8.Font = new Font("Yu Gothic UI", 15F, FontStyle.Regular, GraphicsUnit.Point);
            button8.Location = new Point(626, 306);
            button8.Name = "button8";
            button8.Size = new Size(183, 69);
            button8.TabIndex = 3;
            button8.Text = "計測開始";
            button8.UseVisualStyleBackColor = true;
            button8.Click += button8_Click;
            // 
            // label2
            // 
            label2.AutoSize = true;
            label2.Location = new Point(582, 287);
            label2.Name = "label2";
            label2.Size = new Size(54, 46);
            label2.TabIndex = 9;
            label2.Text = "②";
            // 
            // textBox1
            // 
            textBox1.Location = new Point(626, 118);
            textBox1.Margin = new Padding(3, 4, 3, 4);
            textBox1.Name = "textBox1";
            textBox1.ReadOnly = true;
            textBox1.Size = new Size(232, 52);
            textBox1.TabIndex = 8;
            textBox1.TextChanged += textBox1_TextChanged_4;
            // 
            // label1
            // 
            label1.AutoSize = true;
            label1.Location = new Point(562, 68);
            label1.Name = "label1";
            label1.Size = new Size(156, 46);
            label1.TabIndex = 6;
            label1.Text = "計測距離";
            label1.Click += label1_Click;
            // 
            // button7
            // 
            button7.Font = new Font("Yu Gothic UI", 15F, FontStyle.Regular, GraphicsUnit.Point);
            button7.Location = new Point(626, 215);
            button7.Name = "button7";
            button7.Size = new Size(183, 69);
            button7.TabIndex = 2;
            button7.Text = "基準点の設定";
            button7.UseVisualStyleBackColor = true;
            button7.UseWaitCursor = true;
            button7.Click += button7_Click;
            // 
            // pictureBox2
            // 
            pictureBox2.Location = new Point(6, 68);
            pictureBox2.Name = "pictureBox2";
            pictureBox2.Size = new Size(533, 381);
            pictureBox2.TabIndex = 1;
            pictureBox2.TabStop = false;
            pictureBox2.Click += pictureBox2_Click;
            // 
            // labelMeasurement
            // 
            labelMeasurement.AutoSize = true;
            labelMeasurement.Location = new Point(582, 192);
            labelMeasurement.Name = "labelMeasurement";
            labelMeasurement.Size = new Size(54, 46);
            labelMeasurement.TabIndex = 0;
            labelMeasurement.Text = "①";
            // 
            // imageList1
            // 
            imageList1.ColorDepth = ColorDepth.Depth8Bit;
            imageList1.ImageSize = new Size(16, 16);
            imageList1.TransparentColor = Color.Transparent;
            // 
            // Form1
            // 
            AutoScaleDimensions = new SizeF(8F, 20F);
            AutoScaleMode = AutoScaleMode.Font;
            ClientSize = new Size(914, 600);
            Controls.Add(calibration);
            Controls.Add(button3);
            Controls.Add(button1);
            Margin = new Padding(3, 4, 3, 4);
            Name = "Form1";
            Text = "Form1";
            Load += Form1_Load;
            calibration.ResumeLayout(false);
            キャリブレーション.ResumeLayout(false);
            キャリブレーション.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)pictureBox1).EndInit();
            計測.ResumeLayout(false);
            計測.PerformLayout();
            ((System.ComponentModel.ISupportInitialize)pictureBox2).EndInit();
            ResumeLayout(false);
        }

        #endregion

        private Button button1;
        private Button button3;
        private TabControl calibration;
        private TabPage キャリブレーション;
        private TabPage 計測;
        public ImageList imageList1;
        public Button button2;
        public PictureBox pictureBox1;
        public ListBox listBox1;
        private Button button5;
        public Button button4;
        public PictureBox pictureBox2;
        public ProgressBar progressBar2;
        private Button button8;
        private Button button7;
        public TextBox textBox1;
        private Label label1;
        public Button button6;
        private Label label2;
    }
}