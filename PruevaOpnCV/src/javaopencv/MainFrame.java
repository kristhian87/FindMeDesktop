package javaopencv;

import java.awt.BorderLayout;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.InputStream;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.Iterator;
import java.util.List;
import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.highgui.VideoCapture;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.video.BackgroundSubtractorMOG;

public class MainFrame extends javax.swing.JFrame {  

    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    private Boolean empezar = false;
    private Boolean primerFrame = true;
    private VideoCapture video = null;
    private CamUno camUno = null;
    private CamDos camDos = null;
    private MatOfByte matOfByte = new MatOfByte();
    private BufferedImage bufImagen = null;
    private InputStream in;
    private Mat frameaux = new Mat();
    private Mat frame = new Mat(240, 320, CvType.CV_8UC3);
    private Mat ultimoFrame = new Mat(240, 320, CvType.CV_8UC3);
    private Mat frameActual = new Mat(240, 320, CvType.CV_8UC3);
    private Mat frameProcesado = new Mat(240, 320, CvType.CV_8UC3);
    private ImagePanel imagen;
    private BackgroundSubtractorMOG bsMOG = new BackgroundSubtractorMOG();
    private int retardo = 0;
    String actualDir = "";
    String deteccionDir = "detecciones";
    CascadeClassifier faceDetector = new CascadeClassifier(MainFrame.class.getResource("haarcascade_frontalface_alt.xml").getPath().substring(1));
    MatOfRect rostrosDetectados = new MatOfRect();

    public MainFrame() {
        initComponents();
        jTextFieldSource1.setText("0");
        jButtonStart.setText("Iniciar");
        jButtonStop.setText("Detener");
        jLabelSource1.setText("Camara Funte");
        jCheckBoxFaceDtc.setText("Detectar");
        jCheckBoxMotionDetection.setText("Dtr Movimiento");
        jCheckBoxSave.setText("Guardar Captura");
        jLabel1.setText("Límite:");
        jLabel3.setText("Sensibilidad:");
        jCheckBoxAlarm.setText("Alerta");
        jLabel2.setText("Zero para Webcam local");
        imagen = new ImagePanel(new ImageIcon("figs/320x240.gif").getImage());
        jPanelSource1.add(imagen, BorderLayout.CENTER);
        this.setTitle("Pueva OpenCV");
        actualDir = Paths.get(".").toAbsolutePath().normalize().toString();
        deteccionDir = actualDir + File.separator + deteccionDir;
        // new java.io.File( "." ).getCanonicalPath();
        System.out.println("Actual dir: " + actualDir);
        System.out.println("Detecciones dir: " + deteccionDir);
        jTextFieldSaveLocation.setText(deteccionDir);
    }

    private void start() {
        System.out.println("Presionaste el boton de iniciar!");

        if (!empezar) {
            int sourcen = Integer.parseInt(jTextFieldSource1.getText());
            System.out.println("Abriendo fuente: " + sourcen);

            video = new VideoCapture(sourcen);

            if (video.isOpened()) {
                camUno = new CamUno();
                camUno.start();
                camDos = new CamDos();
                camDos.start();
                empezar = true;
                primerFrame = true;
            }
        }
    }

    private void stop() {
        //System.out.println("You clicked the stop button!");

        if (empezar) {
            try {
                Thread.sleep(500);
            } catch (Exception ex) {
            }
            video.release();
            empezar = false;
        }
    }

    public static String getCurrentTimeStamp() {
        SimpleDateFormat sdfFecha = new SimpleDateFormat("yyyy-MM-dd_HH-mm-ss");//dd/MM/yyyy
        Date fechaActual = new Date();
        String strFecha = sdfFecha.format(fechaActual);
        return strFecha;
    }

    public ArrayList<Rect> deteccion_contornos(Mat frame, Mat outmat) {
        Mat v = new Mat();
        Mat vv = outmat.clone();
        List<MatOfPoint> contornos = new ArrayList();
        Imgproc.findContours(vv, contornos, v, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);

        double maxArea = 100;
        int maxAreaIdx;
        Rect r;
        ArrayList<Rect> rect_array = new ArrayList();

        for (int idx = 0; idx < contornos.size(); idx++) {
            Mat contorno = contornos.get(idx);
            double contornoarea = Imgproc.contourArea(contorno);
            if (contornoarea > maxArea) {
                // maxArea = contornoarea;
                maxAreaIdx = idx;
                r = Imgproc.boundingRect(contornos.get(maxAreaIdx));
                rect_array.add(r);
                Imgproc.drawContours(frame, contornos, maxAreaIdx, new Scalar(0, 0, 255));
            }
        }

        v.release();
        return rect_array;
    }

    class CamUno extends Thread {

        @Override
        public void run() {
            if (video.isOpened()) {
                while (empezar == true) {
                    video.read(frameaux);
                    video.retrieve(frameaux);
                    Imgproc.resize(frameaux, frame, frame.size());
                    frame.copyTo(frameActual);

                    if (primerFrame) {
                        frame.copyTo(ultimoFrame);
                        primerFrame = false;
                        continue;
                    }
                   

                    if (jCheckBoxFaceDtc.isSelected()) {
                        faceDetector.detectMultiScale(frameActual, rostrosDetectados);
                        System.out.println(String.format("Detected %s faces", rostrosDetectados.toArray().length));
                        for (Rect rect : rostrosDetectados.toArray()) {
                            Core.rectangle(frameActual, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(255, 100, 0));
                        }
                        if (jCheckBoxSave.isSelected()) {
                            if (retardo == 2) {
                                String filename = jTextFieldSaveLocation.getText() + File.separator + "capture_" + getCurrentTimeStamp() + ".jpg";
                                System.out.println("Saving results in: " + filename);
                                Highgui.imwrite(filename, frameProcesado);
                                retardo = 0;
                            } else {
                                retardo = retardo + 1;
                            }
                        }
                    }

                    if (jCheckBoxMotionDetection.isSelected()) {

                        Imgproc.GaussianBlur(frameActual, frameActual, new Size(3, 3), 0);
                        Imgproc.GaussianBlur(ultimoFrame, ultimoFrame, new Size(3, 3), 0);

                        //bsMOG.apply(frame, frameProcesado, 0.005);
                        Core.subtract(frameActual, ultimoFrame, frameProcesado);
                        //Core.absdiff(frame,ultimoFrame,frameProcesado);

                        Imgproc.cvtColor(frameProcesado, frameProcesado, Imgproc.COLOR_RGB2GRAY);
                        //

                        int threshold = jSliderThreshold.getValue();
                        //Imgproc.adaptiveThreshold(frameProcesado, frameProcesado, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY_INV, 5, 2);
                        Imgproc.threshold(frameProcesado, frameProcesado, threshold, 255, Imgproc.THRESH_BINARY);

                        ArrayList<Rect> array = deteccion_contornos(frameActual, frameProcesado);
                        ///

                        if (array.size() > 0) {
                            Iterator<Rect> it2 = array.iterator();
                            while (it2.hasNext()) {
                                Rect obj = it2.next();
                                Core.rectangle(frameActual, obj.br(), obj.tl(),
                                        new Scalar(0, 255, 0), 1);
                            }
                        }
                        if (jCheckBoxAlarm.isSelected()) {
                            double sensibility = jSliderSensibility.getValue();
                            //System.out.println(sensibility);
                            double nonZeroPixels = Core.countNonZero(frameProcesado);
                            //System.out.println("nonZeroPixels: " + nonZeroPixels);

                            double nrows = frameProcesado.rows();
                            double ncols = frameProcesado.cols();
                            double total = nrows * ncols / 10;

                            double detections = (nonZeroPixels / total) * 100;
                            //System.out.println(detections);
                            if (detections >= sensibility) {
                                //System.out.println("ALARM ENABLED!");
                                Core.putText(frameActual, "MOTION DETECTED",
                                        new Point(5, frameActual.cols() / 2), //frameActual.rows()/2 frameActual.cols()/2
                                        Core.FONT_HERSHEY_TRIPLEX, new Double(1), new Scalar(0, 0, 255));

                                if (jCheckBoxSave.isSelected()) {
                                    if (retardo == 1000) {
                                        String filename = jTextFieldSaveLocation.getText() + File.separator + "capture_" + getCurrentTimeStamp() + ".jpg";
                                        System.out.println("Saving results in: " + filename);
                                        Highgui.imwrite(filename, frameProcesado);
                                        retardo = 0;
                                    } else {
                                        retardo = retardo + 1;
                                    }
                                }
                            } else {
                                retardo = 0;
                                //System.out.println("");
                            }
                        }
                        //frameActual.copyTo(frameProcesado);
                    } else {
                        //frame.copyTo(frameProcesado);
                    }

                    frameActual.copyTo(frameProcesado);

                    Highgui.imencode(".jpg", frameProcesado, matOfByte);
                    byte[] byteArray = matOfByte.toArray();
                    try {
                        in = new ByteArrayInputStream(byteArray);
                        bufImagen = ImageIO.read(in);
                    } catch (Exception ex) {
                        ex.printStackTrace();
                    }
                    imagen.updateImage(new ImageIcon("figs/lena.png").getImage());
                    imagen.updateImage(bufImagen);

                    frame.copyTo(ultimoFrame);

                    try {
                        Thread.sleep(1);
                    } catch (Exception ex) {
                    }
                }
            }
        }
    }
    class CamDos extends Thread {

        @Override
        public void run() {
            if (video.isOpened()) {
                while (empezar == true) {
                    video.read(frameaux);
                    video.retrieve(frameaux);
                    Imgproc.resize(frameaux, frame, frame.size());
                    frame.copyTo(frameActual);

                    if (primerFrame) {
                        frame.copyTo(ultimoFrame);
                        primerFrame = false;
                        continue;
                    }
                   

                    if (jCheckBoxFaceDtc.isSelected()) {
                        faceDetector.detectMultiScale(frameActual, rostrosDetectados);
                        System.out.println(String.format("Detected %s faces", rostrosDetectados.toArray().length));
                        for (Rect rect : rostrosDetectados.toArray()) {
                            Core.rectangle(frameActual, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(255, 100, 0));
                        }
                        if (jCheckBoxSave.isSelected()) {
                            if (retardo == 2) {
                                String filename = jTextFieldSaveLocation.getText() + File.separator + "capture_" + getCurrentTimeStamp() + ".jpg";
                                System.out.println("Saving results in: " + filename);
                                Highgui.imwrite(filename, frameProcesado);
                                retardo = 0;
                            } else {
                                retardo = retardo + 1;
                            }
                        }
                    }

                    if (jCheckBoxMotionDetection.isSelected()) {

                        Imgproc.GaussianBlur(frameActual, frameActual, new Size(3, 3), 0);
                        Imgproc.GaussianBlur(ultimoFrame, ultimoFrame, new Size(3, 3), 0);

                        //bsMOG.apply(frame, frameProcesado, 0.005);
                        Core.subtract(frameActual, ultimoFrame, frameProcesado);
                        //Core.absdiff(frame,ultimoFrame,frameProcesado);

                        Imgproc.cvtColor(frameProcesado, frameProcesado, Imgproc.COLOR_RGB2GRAY);
                        //

                        int threshold = jSliderThreshold.getValue();
                        //Imgproc.adaptiveThreshold(frameProcesado, frameProcesado, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY_INV, 5, 2);
                        Imgproc.threshold(frameProcesado, frameProcesado, threshold, 255, Imgproc.THRESH_BINARY);

                        ArrayList<Rect> array = deteccion_contornos(frameActual, frameProcesado);
                        ///

                        if (array.size() > 0) {
                            Iterator<Rect> it2 = array.iterator();
                            while (it2.hasNext()) {
                                Rect obj = it2.next();
                                Core.rectangle(frameActual, obj.br(), obj.tl(),
                                        new Scalar(0, 255, 0), 1);
                            }
                        }
                        if (jCheckBoxAlarm.isSelected()) {
                            double sensibility = jSliderSensibility.getValue();
                            //System.out.println(sensibility);
                            double nonZeroPixels = Core.countNonZero(frameProcesado);
                            //System.out.println("nonZeroPixels: " + nonZeroPixels);

                            double nrows = frameProcesado.rows();
                            double ncols = frameProcesado.cols();
                            double total = nrows * ncols / 10;

                            double detections = (nonZeroPixels / total) * 100;
                            //System.out.println(detections);
                            if (detections >= sensibility) {
                                //System.out.println("ALARM ENABLED!");
                                Core.putText(frameActual, "MOTION DETECTED",
                                        new Point(5, frameActual.cols() / 2), //frameActual.rows()/2 frameActual.cols()/2
                                        Core.FONT_HERSHEY_TRIPLEX, new Double(1), new Scalar(0, 0, 255));

                                if (jCheckBoxSave.isSelected()) {
                                    if (retardo == 1000) {
                                        String filename = jTextFieldSaveLocation.getText() + File.separator + "capture_" + getCurrentTimeStamp() + ".jpg";
                                        System.out.println("Saving results in: " + filename);
                                        Highgui.imwrite(filename, frameProcesado);
                                        retardo = 0;
                                    } else {
                                        retardo = retardo + 1;
                                    }
                                }
                            } else {
                                retardo = 0;
                                //System.out.println("");
                            }
                        }
                        //frameActual.copyTo(frameProcesado);
                    } else {
                        //frame.copyTo(frameProcesado);
                    }

                    frameActual.copyTo(frameProcesado);

                    Highgui.imencode(".jpg", frameProcesado, matOfByte);
                    byte[] byteArray = matOfByte.toArray();
                    try {
                        in = new ByteArrayInputStream(byteArray);
                        bufImagen = ImageIO.read(in);
                    } catch (Exception ex) {
                        ex.printStackTrace();
                    }
                    imagen.updateImage(new ImageIcon("figs/lena.png").getImage());
                    imagen.updateImage(bufImagen);

                    frame.copyTo(ultimoFrame);

                    try {
                        Thread.sleep(1);
                    } catch (Exception ex) {
                    }
                }
            }
        }
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        jPanelSource1 = new javax.swing.JPanel();
        jLabelSource1 = new javax.swing.JLabel();
        jButtonStart = new javax.swing.JButton();
        jButtonStop = new javax.swing.JButton();
        jCheckBoxMotionDetection = new javax.swing.JCheckBox();
        jSliderThreshold = new javax.swing.JSlider();
        jLabel1 = new javax.swing.JLabel();
        jLabel2 = new javax.swing.JLabel();
        jCheckBoxAlarm = new javax.swing.JCheckBox();
        jLabel3 = new javax.swing.JLabel();
        jSliderSensibility = new javax.swing.JSlider();
        jTextFieldSaveLocation = new javax.swing.JTextField();
        jCheckBoxSave = new javax.swing.JCheckBox();
        jTextFieldSource1 = new javax.swing.JTextField();
        jCheckBoxFaceDtc = new javax.swing.JCheckBox();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);
        setTitle("Java OpenCV Webcam");

        jPanelSource1.setBorder(javax.swing.BorderFactory.createLineBorder(new java.awt.Color(0, 0, 0)));

        javax.swing.GroupLayout jPanelSource1Layout = new javax.swing.GroupLayout(jPanelSource1);
        jPanelSource1.setLayout(jPanelSource1Layout);
        jPanelSource1Layout.setHorizontalGroup(
            jPanelSource1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 318, Short.MAX_VALUE)
        );
        jPanelSource1Layout.setVerticalGroup(
            jPanelSource1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 238, Short.MAX_VALUE)
        );

        jLabelSource1.setHorizontalAlignment(javax.swing.SwingConstants.CENTER);
        jLabelSource1.setText("Source 1:");

        jButtonStart.setText("Start");
        jButtonStart.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButtonStartActionPerformed(evt);
            }
        });

        jButtonStop.setText("Stop");
        jButtonStop.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButtonStopActionPerformed(evt);
            }
        });

        jCheckBoxMotionDetection.setText("Motion Detection");

        jSliderThreshold.setMaximum(255);
        jSliderThreshold.setPaintLabels(true);
        jSliderThreshold.setPaintTicks(true);
        jSliderThreshold.setValue(15);

        jLabel1.setText("Threshold:");

        jLabel2.setText("(zero for local webcamera)");

        jCheckBoxAlarm.setText("Alarm");

        jLabel3.setText("Sensibility:");

        jSliderSensibility.setMinimum(1);
        jSliderSensibility.setPaintLabels(true);
        jSliderSensibility.setPaintTicks(true);
        jSliderSensibility.setValue(10);

        jCheckBoxSave.setText("Save detections in:");

        jTextFieldSource1.setText("0");

        jCheckBoxFaceDtc.setText("Face Detection");

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
                        .addGroup(layout.createSequentialGroup()
                            .addComponent(jButtonStart, javax.swing.GroupLayout.PREFERRED_SIZE, 80, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                            .addComponent(jButtonStop, javax.swing.GroupLayout.PREFERRED_SIZE, 80, javax.swing.GroupLayout.PREFERRED_SIZE))
                        .addGroup(layout.createSequentialGroup()
                            .addComponent(jCheckBoxSave)
                            .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.UNRELATED)
                            .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
                                .addComponent(jTextFieldSaveLocation, javax.swing.GroupLayout.PREFERRED_SIZE, 195, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addGroup(layout.createSequentialGroup()
                                    .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING, false)
                                        .addComponent(jLabel1, javax.swing.GroupLayout.Alignment.LEADING)
                                        .addComponent(jLabel3))
                                    .addGap(18, 18, 18)
                                    .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                                        .addComponent(jSliderThreshold, javax.swing.GroupLayout.PREFERRED_SIZE, 100, javax.swing.GroupLayout.PREFERRED_SIZE)
                                        .addComponent(jSliderSensibility, javax.swing.GroupLayout.PREFERRED_SIZE, 100, javax.swing.GroupLayout.PREFERRED_SIZE))))))
                    .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
                        .addComponent(jCheckBoxFaceDtc, javax.swing.GroupLayout.Alignment.LEADING)
                        .addComponent(jCheckBoxMotionDetection, javax.swing.GroupLayout.Alignment.LEADING)
                        .addComponent(jCheckBoxAlarm, javax.swing.GroupLayout.Alignment.LEADING)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(jPanelSource1, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addGroup(layout.createSequentialGroup()
                                .addComponent(jLabelSource1)
                                .addGap(18, 18, 18)
                                .addComponent(jTextFieldSource1, javax.swing.GroupLayout.PREFERRED_SIZE, 34, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addGap(36, 36, 36)
                                .addComponent(jLabel2)))))
                .addContainerGap(27, Short.MAX_VALUE))
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(jLabelSource1)
                    .addComponent(jLabel2)
                    .addComponent(jTextFieldSource1, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                .addComponent(jPanelSource1, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, 22, Short.MAX_VALUE)
                .addComponent(jCheckBoxFaceDtc)
                .addGap(10, 10, 10)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addComponent(jLabel1)
                    .addGroup(layout.createSequentialGroup()
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(jCheckBoxMotionDetection)
                            .addComponent(jSliderThreshold, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                        .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(jSliderSensibility, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                                .addComponent(jCheckBoxAlarm)
                                .addComponent(jLabel3)))))
                .addGap(18, 18, 18)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(jTextFieldSaveLocation, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jCheckBoxSave))
                .addGap(18, 18, 18)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(jButtonStart)
                    .addComponent(jButtonStop))
                .addContainerGap())
        );

        pack();
    }// </editor-fold>//GEN-END:initComponents

    private void jButtonStartActionPerformed(java.awt.event.ActionEvent evt)//GEN-FIRST:event_jButtonStartActionPerformed
    {//GEN-HEADEREND:event_jButtonStartActionPerformed

        start();
    }//GEN-LAST:event_jButtonStartActionPerformed

    private void jButtonStopActionPerformed(java.awt.event.ActionEvent evt)//GEN-FIRST:event_jButtonStopActionPerformed
    {//GEN-HEADEREND:event_jButtonStopActionPerformed
        stop();
    }//GEN-LAST:event_jButtonStopActionPerformed

    public static void main(String args[]) {
        /* Set the Nimbus look and feel */
        //<editor-fold defaultstate="collapsed" desc=" Look and feel setting code (optional) ">
        /* If Nimbus (introduced in Java SE 6) is not available, stay with the default look and feel.
     * For details see http://download.oracle.com/javase/tutorial/uiswing/lookandfeel/plaf.html 
         */
        try {
            for (javax.swing.UIManager.LookAndFeelInfo info : javax.swing.UIManager.getInstalledLookAndFeels()) {
                if ("Windows".equals(info.getName())) {
                    javax.swing.UIManager.setLookAndFeel(info.getClassName());
                    break;

                }
            }
        } catch (ClassNotFoundException ex) {
            java.util.logging.Logger.getLogger(MainFrame.class
                    .getName()).log(java.util.logging.Level.SEVERE, null, ex);

        } catch (InstantiationException ex) {
            java.util.logging.Logger.getLogger(MainFrame.class
                    .getName()).log(java.util.logging.Level.SEVERE, null, ex);

        } catch (IllegalAccessException ex) {
            java.util.logging.Logger.getLogger(MainFrame.class
                    .getName()).log(java.util.logging.Level.SEVERE, null, ex);

        } catch (javax.swing.UnsupportedLookAndFeelException ex) {
            java.util.logging.Logger.getLogger(MainFrame.class
                    .getName()).log(java.util.logging.Level.SEVERE, null, ex);
        }
        //</editor-fold>

        /* Create and display the form */
        java.awt.EventQueue.invokeLater(new Runnable() {
            public void run() {
                MainFrame mainFrame = new MainFrame();
                mainFrame.setVisible(true);
                mainFrame.setLocationRelativeTo(null);
            }
        });
    }

    // Variables declaration - do not modify//GEN-BEGIN:variables
    private javax.swing.JButton jButtonStart;
    private javax.swing.JButton jButtonStop;
    private javax.swing.JCheckBox jCheckBoxAlarm;
    private javax.swing.JCheckBox jCheckBoxFaceDtc;
    private javax.swing.JCheckBox jCheckBoxMotionDetection;
    private javax.swing.JCheckBox jCheckBoxSave;
    private javax.swing.JLabel jLabel1;
    private javax.swing.JLabel jLabel2;
    private javax.swing.JLabel jLabel3;
    private javax.swing.JLabel jLabelSource1;
    private javax.swing.JPanel jPanelSource1;
    private javax.swing.JSlider jSliderSensibility;
    private javax.swing.JSlider jSliderThreshold;
    private javax.swing.JTextField jTextFieldSaveLocation;
    private javax.swing.JTextField jTextFieldSource1;
    // End of variables declaration//GEN-END:variables
}
