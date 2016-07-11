package Principal;

import java.awt.BorderLayout;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.InputStream;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.file.Paths;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.Iterator;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
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
    private VideoCapture video1 = null;
    private VideoCapture video2 = null;
    private CamUno camUno = null;
    private CamDos camDos = null;
    private MatOfByte matOfByte1 = new MatOfByte();
    private BufferedImage bufImagen1 = null;
    private InputStream in1;
    private Mat frameaux1 = new Mat();
    private Mat frame1 = new Mat(240, 320, CvType.CV_8UC3);
    private Mat ultimoFrame1 = new Mat(240, 320, CvType.CV_8UC3);
    private Mat frameActual1 = new Mat(240, 320, CvType.CV_8UC3);
    private Mat frameProcesado1 = new Mat(240, 320, CvType.CV_8UC3);
     private MatOfByte matOfByte2 = new MatOfByte();
    private BufferedImage bufImagen2 = null;
    private InputStream in2;
    private Mat frameaux2 = new Mat();
    private Mat frame2 = new Mat(240, 320, CvType.CV_8UC3);
    private Mat ultimoFrame2 = new Mat(240, 320, CvType.CV_8UC3);
    private Mat frameActual2 = new Mat(240, 320, CvType.CV_8UC3);
    private Mat frameProcesado2 = new Mat(240, 320, CvType.CV_8UC3);
    private ImagePanel imagen1;
    private ImagePanel imagen2;
    private BackgroundSubtractorMOG bsMOG = new BackgroundSubtractorMOG();
    private int retardo = 0;
    String actualDir = "";
    String deteccionDir = "detecciones";
    //CascadeClassifier faceDetector = new CascadeClassifier(getClass.getResource("haarcascade_frontalface_alt.xml").getPath().substring(1));

    MatOfRect rostrosDetectados = new MatOfRect();

    public MainFrame() {
        initComponents();
        jTextFieldSource1.setText("0");
        jTextFieldSource2.setText("2");
        jButtonStart.setText("Iniciar");
        jButtonStop.setText("Detener");
        jLabelSource1.setText("Camara Funte");
        jLabelSource2.setText("Camara Funte");
        jCheckBoxFaceDtc.setText("Detectar");
        jCheckBoxMotionDetection.setText("Dtr Movimiento");
        jCheckBoxSave.setText("Guardar Captura");
        jLabel1.setText("Límite:");
        jLabel3.setText("Sensibilidad:");
        jCheckBoxAlarm.setText("Alerta");
        jLabel2.setText("Zero para Webcam local");
        imagen1 = new ImagePanel(new ImageIcon("figs/320x240.gif").getImage());
        imagen2 = new ImagePanel(new ImageIcon("figs/320x240.gif").getImage());
        jPanelSource1.add(imagen1, BorderLayout.CENTER);
        jPanelSource2.add(imagen2, BorderLayout.CENTER);
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
            int sourcen1 = Integer.parseInt(jTextFieldSource1.getText());
            System.out.println("Abriendo fuente: " + sourcen1);
            int sourcen2 = Integer.parseInt(jTextFieldSource2.getText());
            System.out.println("Abriendo fuente: " + sourcen2);

            video1 = new VideoCapture(sourcen1);
            video2 = new VideoCapture(sourcen2);
            

            if (video1.isOpened()) {
                camUno = new CamUno();
                camUno.start();
                empezar = true;
                primerFrame = true;
            }
            if (video2.isOpened()) {
                camDos = new CamDos();
                camDos.start();
                empezar = true;
                primerFrame = true;
            }
        }
    }

    private void stop() {
        System.out.println("Presionaste el boton de iniciar!");

        if (empezar) {
            try {
                Thread.sleep(500);
            } catch (Exception ex) {
            }
            video1.release();
            video2.release();
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
            try {
                final URI xmlUri = getClass().getResource("/Patrones/haarcascade_frontalface_alt.xml").toURI();
                //System.out.println(getClass().getResource("/lbpcascade_frontalface.xml"));
                final CascadeClassifier faceDetector = new CascadeClassifier(new File(xmlUri).getAbsolutePath());
                if (video1.isOpened()) {
                    while (empezar == true) {
                        video1.read(frameaux1);
                        video1.retrieve(frameaux1);
                        Imgproc.resize(frameaux1, frame1, frame1.size());
                        frame1.copyTo(frameActual1);

                        if (primerFrame) {
                            frame1.copyTo(ultimoFrame1);
                            primerFrame = false;
                            continue;
                        }

                        if (jCheckBoxFaceDtc.isSelected()) {
                            faceDetector.detectMultiScale(frameActual1, rostrosDetectados);
                            System.out.println(String.format("Detected %s faces", rostrosDetectados.toArray().length));
                            for (Rect rect : rostrosDetectados.toArray()) {
                                Core.rectangle(frameActual1, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(255, 100, 0));
                            }
                            if (jCheckBoxSave.isSelected()) {
                                if (retardo == 2) {
                                    String filename = jTextFieldSaveLocation.getText() + File.separator + "capture_" + getCurrentTimeStamp() + ".jpg";
                                    System.out.println("Saving results in: " + filename);
                                    Highgui.imwrite(filename, frameProcesado1);
                                    retardo = 0;
                                } else {
                                    retardo = retardo + 1;
                                }
                            }
                        }

                        if (jCheckBoxMotionDetection.isSelected()) {

                            Imgproc.GaussianBlur(frameActual1, frameActual1, new Size(3, 3), 0);
                            Imgproc.GaussianBlur(ultimoFrame1, ultimoFrame1, new Size(3, 3), 0);

                            //bsMOG.apply(frame, frameProcesado, 0.005);
                            Core.subtract(frameActual1, ultimoFrame1, frameProcesado1);
                            //Core.absdiff(frame,ultimoFrame,frameProcesado);

                            Imgproc.cvtColor(frameProcesado1, frameProcesado1, Imgproc.COLOR_RGB2GRAY);
                            //

                            int threshold = jSliderThreshold.getValue();
                            //Imgproc.adaptiveThreshold(frameProcesado, frameProcesado, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY_INV, 5, 2);
                            Imgproc.threshold(frameProcesado1, frameProcesado1, threshold, 255, Imgproc.THRESH_BINARY);

                            ArrayList<Rect> array = deteccion_contornos(frameActual1, frameProcesado1);
                            ///

                            if (array.size() > 0) {
                                Iterator<Rect> it2 = array.iterator();
                                while (it2.hasNext()) {
                                    Rect obj = it2.next();
                                    Core.rectangle(frameActual1, obj.br(), obj.tl(),
                                            new Scalar(0, 255, 0), 1);
                                }
                            }
                            if (jCheckBoxAlarm.isSelected()) {
                                double sensibility = jSliderSensibility.getValue();
                                //System.out.println(sensibility);
                                double nonZeroPixels = Core.countNonZero(frameProcesado1);
                                //System.out.println("nonZeroPixels: " + nonZeroPixels);

                                double nrows = frameProcesado1.rows();
                                double ncols = frameProcesado1.cols();
                                double total = nrows * ncols / 10;

                                double detections = (nonZeroPixels / total) * 100;
                                //System.out.println(detections);
                                if (detections >= sensibility) {
                                    //System.out.println("ALARM ENABLED!");
                                    Core.putText(frameActual1, "MOTION DETECTED",
                                            new Point(5, frameActual1.cols() / 2), //frameActual.rows()/2 frameActual.cols()/2
                                            Core.FONT_HERSHEY_TRIPLEX, new Double(1), new Scalar(0, 0, 255));

                                    if (jCheckBoxSave.isSelected()) {
                                        if (retardo == 1000) {
                                            String filename = jTextFieldSaveLocation.getText() + File.separator + "capture_" + getCurrentTimeStamp() + ".jpg";
                                            System.out.println("Saving results in: " + filename);
                                            Highgui.imwrite(filename, frameProcesado1);
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

                        frameActual1.copyTo(frameProcesado1);

                        Highgui.imencode(".jpg", frameProcesado1, matOfByte1);
                        byte[] byteArray = matOfByte1.toArray();
                        try {
                            in1 = new ByteArrayInputStream(byteArray);
                            bufImagen1 = ImageIO.read(in1);
                        } catch (Exception ex) {
                            ex.printStackTrace();
                        }
                        imagen1.updateImage(new ImageIcon("figs/lena.png").getImage());
                        imagen1.updateImage(bufImagen1);

                        frame1.copyTo(ultimoFrame1);

                        try {
                            Thread.sleep(1);
                        } catch (Exception ex) {
                        }
                    }
                }
            } catch (URISyntaxException ex) {
                Logger.getLogger(MainFrame.class.getName()).log(Level.SEVERE, null, ex);
            }
        }
    }

    class CamDos extends Thread {

        @Override
        public void run() {
            try {
                final URI xmlUri = getClass().getResource("/Patrones/haarcascade_frontalface_alt.xml").toURI();
                //System.out.println(getClass().getResource("/lbpcascade_frontalface.xml"));
                final CascadeClassifier faceDetector = new CascadeClassifier(new File(xmlUri).getAbsolutePath());
                if (video2.isOpened()) {
                    while (empezar == true) {
                        video2.read(frameaux2);
                        video2.retrieve(frameaux2);
                        Imgproc.resize(frameaux2, frame2, frame2.size());
                        frame2.copyTo(frameActual2);

                        if (primerFrame) {
                            frame2.copyTo(ultimoFrame2);
                            primerFrame = false;
                            continue;
                        }

                        if (jCheckBoxFaceDtc.isSelected()) {
                            faceDetector.detectMultiScale(frameActual2, rostrosDetectados);
                            System.out.println(String.format("Detected %s faces", rostrosDetectados.toArray().length));
                            for (Rect rect : rostrosDetectados.toArray()) {
                                Core.rectangle(frameActual2, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(255, 100, 0));
                            }
                            if (jCheckBoxSave.isSelected()) {
                                if (retardo == 2) {
                                    String filename = jTextFieldSaveLocation.getText() + File.separator + "capture_" + getCurrentTimeStamp() + ".jpg";
                                    System.out.println("Saving results in: " + filename);
                                    Highgui.imwrite(filename, frameProcesado2);
                                    retardo = 0;
                                } else {
                                    retardo = retardo + 1;
                                }
                            }
                        }

                        if (jCheckBoxMotionDetection.isSelected()) {

                            Imgproc.GaussianBlur(frameActual2, frameActual2, new Size(3, 3), 0);
                            Imgproc.GaussianBlur(ultimoFrame2, ultimoFrame2, new Size(3, 3), 0);

                            //bsMOG.apply(frame, frameProcesado, 0.005);
                            Core.subtract(frameActual2, ultimoFrame2, frameProcesado2);
                            //Core.absdiff(frame,ultimoFrame,frameProcesado);

                            Imgproc.cvtColor(frameProcesado2, frameProcesado2, Imgproc.COLOR_RGB2GRAY);
                            //

                            int threshold = jSliderThreshold.getValue();
                            //Imgproc.adaptiveThreshold(frameProcesado, frameProcesado, 255, Imgproc.ADAPTIVE_THRESH_MEAN_C, Imgproc.THRESH_BINARY_INV, 5, 2);
                            Imgproc.threshold(frameProcesado2, frameProcesado2, threshold, 255, Imgproc.THRESH_BINARY);

                            ArrayList<Rect> array = deteccion_contornos(frameActual2, frameProcesado2);
                            ///

                            if (array.size() > 0) {
                                Iterator<Rect> it2 = array.iterator();
                                while (it2.hasNext()) {
                                    Rect obj = it2.next();
                                    Core.rectangle(frameActual2, obj.br(), obj.tl(),
                                            new Scalar(0, 255, 0), 1);
                                }
                            }
                            if (jCheckBoxAlarm.isSelected()) {
                                double sensibility = jSliderSensibility.getValue();
                                //System.out.println(sensibility);
                                double nonZeroPixels = Core.countNonZero(frameProcesado2);
                                //System.out.println("nonZeroPixels: " + nonZeroPixels);

                                double nrows = frameProcesado2.rows();
                                double ncols = frameProcesado2.cols();
                                double total = nrows * ncols / 10;

                                double detections = (nonZeroPixels / total) * 100;
                                //System.out.println(detections);
                                if (detections >= sensibility) {
                                    //System.out.println("ALARM ENABLED!");
                                    Core.putText(frameActual2, "MOTION DETECTED",
                                            new Point(5, frameActual2.cols() / 2), //frameActual.rows()/2 frameActual.cols()/2
                                            Core.FONT_HERSHEY_TRIPLEX, new Double(1), new Scalar(0, 0, 255));

                                    if (jCheckBoxSave.isSelected()) {
                                        if (retardo == 1000) {
                                            String filename = jTextFieldSaveLocation.getText() + File.separator + "capture_" + getCurrentTimeStamp() + ".jpg";
                                            System.out.println("Saving results in: " + filename);
                                            Highgui.imwrite(filename, frameProcesado2);
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

                        frameActual2.copyTo(frameProcesado2);

                        Highgui.imencode(".jpg", frameProcesado2, matOfByte2);
                        byte[] byteArray = matOfByte2.toArray();
                        try {
                            in2 = new ByteArrayInputStream(byteArray);
                            bufImagen2 = ImageIO.read(in2);
                        } catch (Exception ex) {
                            ex.printStackTrace();
                        }
                        imagen2.updateImage(new ImageIcon("figs/lena.png").getImage());
                        imagen2.updateImage(bufImagen2);

                        frame2.copyTo(ultimoFrame2);

                        try {
                            Thread.sleep(1);
                        } catch (Exception ex) {
                        }
                    }
                }
            } catch (URISyntaxException ex) {
                Logger.getLogger(MainFrame.class.getName()).log(Level.SEVERE, null, ex);
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
        jPanelSource2 = new javax.swing.JPanel();
        jPanel1 = new javax.swing.JPanel();
        jCheckBoxMotionDetection = new javax.swing.JCheckBox();
        jCheckBoxAlarm = new javax.swing.JCheckBox();
        jCheckBoxSave = new javax.swing.JCheckBox();
        jTextFieldSaveLocation = new javax.swing.JTextField();
        jSliderSensibility = new javax.swing.JSlider();
        jSliderThreshold = new javax.swing.JSlider();
        jLabel1 = new javax.swing.JLabel();
        jLabel3 = new javax.swing.JLabel();
        jButtonStart = new javax.swing.JButton();
        jButtonStop = new javax.swing.JButton();
        jLabelSource1 = new javax.swing.JLabel();
        jTextFieldSource1 = new javax.swing.JTextField();
        jLabelSource2 = new javax.swing.JLabel();
        jTextFieldSource2 = new javax.swing.JTextField();
        jLabel2 = new javax.swing.JLabel();
        jCheckBoxFaceDtc = new javax.swing.JCheckBox();

        setDefaultCloseOperation(javax.swing.WindowConstants.EXIT_ON_CLOSE);
        setTitle("Java OpenCV Webcam");

        jPanelSource1.setBorder(javax.swing.BorderFactory.createLineBorder(new java.awt.Color(0, 0, 0)));
        jPanelSource1.setPreferredSize(new java.awt.Dimension(320, 240));

        javax.swing.GroupLayout jPanelSource1Layout = new javax.swing.GroupLayout(jPanelSource1);
        jPanelSource1.setLayout(jPanelSource1Layout);
        jPanelSource1Layout.setHorizontalGroup(
            jPanelSource1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 318, Short.MAX_VALUE)
        );
        jPanelSource1Layout.setVerticalGroup(
            jPanelSource1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 0, Short.MAX_VALUE)
        );

        jPanelSource2.setBorder(javax.swing.BorderFactory.createLineBorder(new java.awt.Color(0, 0, 0)));
        jPanelSource2.setPreferredSize(new java.awt.Dimension(320, 240));

        javax.swing.GroupLayout jPanelSource2Layout = new javax.swing.GroupLayout(jPanelSource2);
        jPanelSource2.setLayout(jPanelSource2Layout);
        jPanelSource2Layout.setHorizontalGroup(
            jPanelSource2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 318, Short.MAX_VALUE)
        );
        jPanelSource2Layout.setVerticalGroup(
            jPanelSource2Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGap(0, 238, Short.MAX_VALUE)
        );

        jCheckBoxMotionDetection.setText("jCheckBoxMotionDetection");

        jCheckBoxAlarm.setText("jCheckBoxAlarm");

        jCheckBoxSave.setText("jCheckBoxSave");

        jSliderSensibility.setMinimum(1);
        jSliderSensibility.setPaintLabels(true);
        jSliderSensibility.setPaintTicks(true);
        jSliderSensibility.setValue(10);

        jSliderThreshold.setMaximum(255);
        jSliderThreshold.setPaintLabels(true);
        jSliderThreshold.setPaintTicks(true);
        jSliderThreshold.setValue(15);

        jLabel1.setText("jLabel1");

        jLabel3.setText("jLabel3");

        jButtonStart.setText("jButtonStart");
        jButtonStart.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButtonStartActionPerformed(evt);
            }
        });

        jButtonStop.setText("jButtonStop");
        jButtonStop.addActionListener(new java.awt.event.ActionListener() {
            public void actionPerformed(java.awt.event.ActionEvent evt) {
                jButtonStopActionPerformed(evt);
            }
        });

        jLabelSource1.setHorizontalAlignment(javax.swing.SwingConstants.CENTER);
        jLabelSource1.setText("jLabelSource1");

        jTextFieldSource1.setText("0");

        jLabelSource2.setHorizontalAlignment(javax.swing.SwingConstants.CENTER);
        jLabelSource2.setText("jLabelSource2");

        jTextFieldSource2.setText("0");

        jLabel2.setText("jLabel2");

        jCheckBoxFaceDtc.setText("jCheckBoxFaceDtc");

        javax.swing.GroupLayout jPanel1Layout = new javax.swing.GroupLayout(jPanel1);
        jPanel1.setLayout(jPanel1Layout);
        jPanel1Layout.setHorizontalGroup(
            jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(jPanel1Layout.createSequentialGroup()
                .addGap(21, 21, 21)
                .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(jPanel1Layout.createSequentialGroup()
                        .addComponent(jCheckBoxMotionDetection)
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, 48, Short.MAX_VALUE)
                        .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, jPanel1Layout.createSequentialGroup()
                                .addComponent(jLabel1)
                                .addGap(18, 18, 18)
                                .addComponent(jSliderThreshold, javax.swing.GroupLayout.PREFERRED_SIZE, 100, javax.swing.GroupLayout.PREFERRED_SIZE))
                            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, jPanel1Layout.createSequentialGroup()
                                .addComponent(jLabel3)
                                .addGap(18, 18, 18)
                                .addComponent(jSliderSensibility, javax.swing.GroupLayout.PREFERRED_SIZE, 100, javax.swing.GroupLayout.PREFERRED_SIZE)))
                        .addGap(67, 67, 67)
                        .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.TRAILING)
                            .addGroup(jPanel1Layout.createSequentialGroup()
                                .addComponent(jLabelSource2)
                                .addGap(18, 18, 18)
                                .addComponent(jTextFieldSource2, javax.swing.GroupLayout.PREFERRED_SIZE, 34, javax.swing.GroupLayout.PREFERRED_SIZE))
                            .addGroup(jPanel1Layout.createSequentialGroup()
                                .addComponent(jLabelSource1)
                                .addGap(18, 18, 18)
                                .addComponent(jTextFieldSource1, javax.swing.GroupLayout.PREFERRED_SIZE, 34, javax.swing.GroupLayout.PREFERRED_SIZE)))
                        .addGap(58, 58, 58)
                        .addComponent(jLabel2)
                        .addGap(174, 174, 174))
                    .addGroup(jPanel1Layout.createSequentialGroup()
                        .addComponent(jCheckBoxAlarm)
                        .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                    .addGroup(jPanel1Layout.createSequentialGroup()
                        .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(jCheckBoxFaceDtc)
                            .addGroup(jPanel1Layout.createSequentialGroup()
                                .addComponent(jCheckBoxSave)
                                .addGap(35, 35, 35)
                                .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                                    .addGroup(jPanel1Layout.createSequentialGroup()
                                        .addComponent(jButtonStart, javax.swing.GroupLayout.PREFERRED_SIZE, 80, javax.swing.GroupLayout.PREFERRED_SIZE)
                                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED)
                                        .addComponent(jButtonStop, javax.swing.GroupLayout.PREFERRED_SIZE, 80, javax.swing.GroupLayout.PREFERRED_SIZE))
                                    .addComponent(jTextFieldSaveLocation, javax.swing.GroupLayout.PREFERRED_SIZE, 404, javax.swing.GroupLayout.PREFERRED_SIZE))))
                        .addGap(0, 0, Short.MAX_VALUE))))
        );
        jPanel1Layout.setVerticalGroup(
            jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, jPanel1Layout.createSequentialGroup()
                .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, jPanel1Layout.createSequentialGroup()
                        .addContainerGap(19, Short.MAX_VALUE)
                        .addComponent(jCheckBoxFaceDtc)
                        .addGap(18, 18, 18)
                        .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                            .addComponent(jCheckBoxMotionDetection)
                            .addComponent(jLabel1))
                        .addGap(26, 26, 26))
                    .addGroup(jPanel1Layout.createSequentialGroup()
                        .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                        .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                            .addComponent(jSliderThreshold, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                            .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                                .addComponent(jLabel2)
                                .addComponent(jTextFieldSource1, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                                .addComponent(jLabelSource1)))
                        .addPreferredGap(javax.swing.LayoutStyle.ComponentPlacement.RELATED, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)))
                .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
                    .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                        .addComponent(jCheckBoxAlarm)
                        .addComponent(jLabelSource2)
                        .addComponent(jLabel3)
                        .addComponent(jTextFieldSource2, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                    .addComponent(jSliderSensibility, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE))
                .addGap(18, 18, 18)
                .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(jTextFieldSaveLocation, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                    .addComponent(jCheckBoxSave))
                .addGap(35, 35, 35)
                .addGroup(jPanel1Layout.createParallelGroup(javax.swing.GroupLayout.Alignment.BASELINE)
                    .addComponent(jButtonStart)
                    .addComponent(jButtonStop))
                .addContainerGap())
        );

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(getContentPane());
        getContentPane().setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(javax.swing.GroupLayout.Alignment.TRAILING, layout.createSequentialGroup()
                .addGap(70, 70, 70)
                .addComponent(jPanelSource1, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(66, 66, 66)
                .addComponent(jPanelSource2, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addContainerGap(javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
            .addGroup(layout.createSequentialGroup()
                .addContainerGap()
                .addComponent(jPanel1, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                .addContainerGap())
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addGap(31, 31, 31)
                .addGroup(layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING, false)
                    .addComponent(jPanelSource2, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
                    .addComponent(jPanelSource1, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE))
                .addGap(18, 18, 18)
                .addComponent(jPanel1, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, Short.MAX_VALUE)
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
    private javax.swing.JLabel jLabelSource2;
    private javax.swing.JPanel jPanel1;
    private javax.swing.JPanel jPanelSource1;
    private javax.swing.JPanel jPanelSource2;
    private javax.swing.JSlider jSliderSensibility;
    private javax.swing.JSlider jSliderThreshold;
    private javax.swing.JTextField jTextFieldSaveLocation;
    private javax.swing.JTextField jTextFieldSource1;
    private javax.swing.JTextField jTextFieldSource2;
    // End of variables declaration//GEN-END:variables
}
