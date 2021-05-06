package com.kafka.afinal;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.drawable.BitmapDrawable;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;


import com.chaquo.python.PyObject;
import com.chaquo.python.Python;
import com.chaquo.python.android.AndroidPlatform;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvException;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.ByteArrayOutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Base64;
import java.util.List;

//package com.isaac.models;
//import com.isaac.utils.Filters;







public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {

    CameraBridgeViewBase cameraBridgeViewBase;
    BaseLoaderCallback baseLoaderCallback;
    int counter = 0;
    Bitmap bitmap;
    String imageString="";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);


        cameraBridgeViewBase = (JavaCameraView)findViewById(R.id.CameraView);
        cameraBridgeViewBase.setVisibility(SurfaceView.VISIBLE);
        cameraBridgeViewBase.setCvCameraViewListener(this);

        //System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        baseLoaderCallback = new BaseLoaderCallback(this) {
            @Override
            public void onManagerConnected(int status) {
                super.onManagerConnected(status);

                switch(status){

                    case BaseLoaderCallback.SUCCESS:
                        cameraBridgeViewBase.enableView();
                        break;
                    default:
                        super.onManagerConnected(status);
                        break;
                }


            }

        };




    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Mat frame = inputFrame.rgba();
        bitmap = convertMatToBitMap(frame);
        imageString=getStringImage(bitmap);
        if (! Python.isStarted()) {
            Python.start(new AndroidPlatform(this));
        }
        Python py= Python.getInstance();
        PyObject pyobj= py.getModule("example");
        PyObject obj = pyobj.callAttr("main",  imageString);
        frame=enhance(frame,1,100,0.001);
        String str =obj.toString();
        byte data[] = android.util.Base64.decode(str,0);
        Bitmap bmp = BitmapFactory.decodeByteArray(data,0,data.length);
        Utils.bitmapToMat(bmp, frame);

        Core.rotate(frame, frame, Core.ROTATE_90_CLOCKWISE);



        return frame ;
    }

    private String getStringImage(Bitmap bitmap) {
        ByteArrayOutputStream baos  =new ByteArrayOutputStream();
        bitmap.compress(Bitmap.CompressFormat.PNG,100,baos);
        byte[] imageBytes =baos.toByteArray();
        String encodedImage = android.util.Base64.encodeToString(imageBytes,0);
        return encodedImage;

    }

    private static Bitmap convertMatToBitMap(Mat input){
        Bitmap bmp = null;
        Mat rgb = new Mat();
        Imgproc.cvtColor(input, rgb, Imgproc.COLOR_BGR2RGB);

        try {
            bmp = Bitmap.createBitmap(rgb.cols(), rgb.rows(), Bitmap.Config.ARGB_8888);
            Utils.matToBitmap(rgb, bmp);
        }
        catch (CvException e){
            Log.d("Exception",e.getMessage());
        }
        return bmp;
    }
    public static Mat enhance(Mat image, double krnlRatio, double minAtmosLight, double eps) {
        image.convertTo(image, CvType.CV_32F);
        // extract each color channel
        List<Mat> rgb = new ArrayList<>();
        Core.split(image, rgb);
        Mat rChannel = rgb.get(0);
        Mat gChannel = rgb.get(1);
        Mat bChannel = rgb.get(2);
        int rows = rChannel.rows();
        int cols = rChannel.cols();
        // derive the dark channel from original image
        Mat dc = rChannel.clone();
        for (int i = 0; i < image.rows(); i++) {
            for (int j = 0; j < image.cols(); j++) {
                double min = Math.min(rChannel.get(i, j)[0], Math.min(gChannel.get(i, j)[0], bChannel.get(i, j)[0]));
                dc.put(i, j, min);
            }
        }
        // minimum filter
        int krnlSz = Double.valueOf(Math.max(Math.max(rows * krnlRatio, cols * krnlRatio), 3.0)).intValue();
        Mat kernel = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(krnlSz, krnlSz), new Point(-1, -1));
        Imgproc.erode(dc, dc, kernel);
        // get coarse transmission map
        Mat t = dc.clone();
        Core.subtract(t, new Scalar(255.0), t);
        Core.multiply(t, new Scalar(-1.0), t);
        Core.divide(t, new Scalar(255.0), t);
        // obtain gray scale image
        Mat gray = new Mat();
        Imgproc.cvtColor(image, gray, Imgproc.COLOR_RGB2GRAY);
        Core.divide(gray, new Scalar(255.0), gray);
        // refine transmission map
        int r = krnlSz * 4;
        //t = Filters.GuidedImageFilter(gray, t, r, eps);
        // get minimum atmospheric light
        minAtmosLight = Math.min(minAtmosLight, Core.minMaxLoc(dc).maxVal);
        // dehaze each color channel
        rChannel = dehaze(rChannel, t, minAtmosLight);
        gChannel = dehaze(gChannel, t, minAtmosLight);
        bChannel = dehaze(bChannel, t, minAtmosLight);
        // merge three color channels to a image
        Mat outval = new Mat();
        Core.merge(new ArrayList<>(Arrays.asList(rChannel, gChannel, bChannel)), outval);
        outval.convertTo(outval, CvType.CV_8UC1);
        return outval;
    }

    private static Mat dehaze(Mat channel, Mat t, double minAtmosLight) {
        Mat t_ = new Mat();
        Core.subtract(t, new Scalar(1.0), t_);
        Core.multiply(t_, new Scalar(-1.0 * minAtmosLight), t_);
        Core.subtract(channel, t_, channel);
        Core.divide(channel, t, channel);
        return channel;
    }


    @Override
    public void onCameraViewStarted(int width, int height) {

    }


    @Override
    public void onCameraViewStopped() {

    }


    @Override
    protected void onResume() {
        super.onResume();

        if (!OpenCVLoader.initDebug()){
            Toast.makeText(getApplicationContext(),"There's a problem, yo!", Toast.LENGTH_SHORT).show();
        }

        else
        {
            baseLoaderCallback.onManagerConnected(baseLoaderCallback.SUCCESS);
        }



    }

    @Override
    protected void onPause() {
        super.onPause();
        if(cameraBridgeViewBase!=null){

            cameraBridgeViewBase.disableView();
        }

    }


    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cameraBridgeViewBase!=null){
            cameraBridgeViewBase.disableView();
        }
    }
}