## Learning Report--5/11-5/22

### U-Net初解

<img src="https://upload-images.jianshu.io/upload_images/15646173-7332a2f218e28f8c.png?imageMogr2/auto-orient/strip|imageView2/2/w/1200" alt="img" style="zoom: 33%;" />

U-net前半部分作用是特征提取，后半部分是上采样。在一些文献中也把这样的结构叫做编码器-解码器结构。U-net采用了完全不同的特征融合方式：拼接，U-net采用将特征在channel维度拼接在一起，形成更厚的特征。而FCN融合时使用的对应点相加，并不形成更厚的特征。

#### 一、Labelme制作标签

#### 二、Unet网络调参

网络调参涉及以下几个方面：
 （1）加入BN层
 （2）将最后一层激活函数替换成ReLU
 （3）损失函数替换成mse
 多分类一般最后一层原本是softmax，使用了这个激活函数跑完后，没有达到分割效果，所以替换成了之前做过的图对图项目激活函数，效果就出来了，纯属经验之谈，理论还没有进行验证。多分类的损失函数多是交叉熵，经过验证也是不能达到效果，替换成均方根误差。

#### 三、训练与测试

代码放入code中

另外可见：https://blog.csdn.net/huangshaoyin/article/details/81041184



### 图像分割评价标准

https://blog.csdn.net/qq_42450404/article/details/93048736

文献放入reference中



### 医学3D可视化建模

基于VTK的三维重建

```c++
// 读取文件夹下图片，将图像进行轮廓提取后再进行三维重建
int build3DViewFull()
{     
    vtkSmartPointer<vtkRenderer> aRenderer = vtkSmartPointer<vtkRenderer>::New();
    vtkSmartPointer<vtkRenderWindow> renWin = vtkSmartPointer<vtkRenderWindow>::New();
    renWin->AddRenderer(aRenderer);

    vtkSmartPointer<vtkRenderWindowInteractor> iren =
        vtkSmartPointer<vtkRenderWindowInteractor>::New();
    iren->SetRenderWindow(renWin);

    // 新建文件读取对象，常见的有vtkBMPReader、vtkDICOMImageReader、vtkJPEGReader等
    vtkSmartPointer<vtkJPEGReader> jpegReader =
        vtkSmartPointer<vtkJPEGReader>::New();  
    // 不同的reader需要设置的参数是不同的 因此本例仅适合jpegreader
    jpegReader->SetFilePrefix("C:/Users/DawnWind/Desktop/000/"); // 要打开的路径
    jpegReader->SetFilePattern("%s%d.jpg"); // 图片文件名格式，此处为 0.jpg 1.jpg ...
    jpegReader->SetDataByteOrderToLittleEndian();
    jpegReader->SetDataSpacing(1, 1, 1.4);  // 设置图片中像素比，我理解得不清楚，具体请百度之
    jpegReader->SetFileNameSliceSpacing(1); 

    jpegReader->SetDataExtent(0, 209, 0, 209, 0, 29);
    // 这里因为在000文件夹里面有0.jpg ~ 29.jpg，所以设置为 0，29
    // 每张图片的长宽为210 * 210 因此设置为0，209

    jpegReader->Update();  
    // update这里要注意一下，对于VTK在默认情况下是在最后操作时候才一次性刷新
    // 也就是说如果没有自动刷新的话，在一些中间过程中是无法获得到数据的，因为没update进去
    

    vtkSmartPointer<vtkContourFilter> skinExtractor =
        vtkSmartPointer<vtkContourFilter>::New();
    skinExtractor->SetInputConnection(jpegReader->GetOutputPort());
    skinExtractor->SetValue(200, 100);    //值越大，保留的部分越少。

    //重新计算法向量
    vtkSmartPointer<vtkPolyDataNormals> skinNormals =
        vtkSmartPointer<vtkPolyDataNormals>::New();
    skinNormals->SetInputConnection(skinExtractor->GetOutputPort());
    skinNormals->SetFeatureAngle(60.0);      
    //Specify the angle that defines a sharp edge. 
    //If the difference in angle across neighboring polygons is greater than this value, 
    //the shared edge is considered "sharp". 


    //create triangle strips and/or poly-lines 为了更快的显示速度
    vtkSmartPointer<vtkStripper> skinStripper =        
        vtkSmartPointer<vtkStripper>::New();
    skinStripper->SetInputConnection(skinNormals->GetOutputPort()); 

    vtkSmartPointer<vtkPolyDataMapper> skinMapper =
        vtkSmartPointer<vtkPolyDataMapper>::New();
    skinMapper->SetInputConnection(skinStripper->GetOutputPort());
    skinMapper->ScalarVisibilityOff();    //这样不会带颜色


    vtkSmartPointer<vtkActor> skin =
        vtkSmartPointer<vtkActor>::New();
    skin->SetMapper(skinMapper); 

    // An outline provides context around the data.
    // 一个围绕在物体的立体框，可以先忽略
    /*
    vtkSmartPointer<vtkOutlineFilter> outlineData =
        vtkSmartPointer<vtkOutlineFilter>::New();
    outlineData->SetInputConnection(dicomReader->GetOutputPort());

    vtkSmartPointer<vtkPolyDataMapper> mapOutline =
        vtkSmartPointer<vtkPolyDataMapper>::New();
    mapOutline->SetInputConnection(outlineData->GetOutputPort());

    vtkSmartPointer<vtkActor> outline =
        vtkSmartPointer<vtkActor>::New();
    outline->SetMapper(mapOutline);
    outline->GetProperty()->SetColor(0,0,0);
 
    aRenderer->AddActor(outline);
    */
    // It is convenient to create an initial view of the data. The FocalPoint
    // and Position form a vector direction. Later on (ResetCamera() method)
    // this vector is used to position the camera to look at the data in
    // this direction.
    vtkSmartPointer<vtkCamera> aCamera =
        vtkSmartPointer<vtkCamera>::New();
    aCamera->SetViewUp (0, 0, -1);
    aCamera->SetPosition (0, 1, 0);
    aCamera->SetFocalPoint (0, 0, 0);
    aCamera->ComputeViewPlaneNormal();
    aCamera->Azimuth(30.0);
    aCamera->Elevation(30.0);

    // Actors are added to the renderer. An initial camera view is created.
    // The Dolly() method moves the camera towards the FocalPoint,
    // thereby enlarging the image.
    aRenderer->AddActor(skin);
    aRenderer->SetActiveCamera(aCamera);
    aRenderer->ResetCamera ();
    aCamera->Dolly(1.5);

    // Set a background color for the renderer and set the size of the
    // render window (expressed in pixels).
    aRenderer->SetBackground(.2, .3, .4);
    renWin->SetSize(640, 480);

    // Note that when camera movement occurs (as it does in the Dolly()
    // method), the clipping planes often need adjusting. Clipping planes
    // consist of two planes: near and far along the view direction. The 
    // near plane clips out objects in front of the plane; the far plane
    // clips out objects behind the plane. This way only what is drawn
    // between the planes is actually rendered.
    aRenderer->ResetCameraClippingRange ();

    // Initialize the event loop and then start it.
    iren->Initialize();
    iren->Start();
    return 0;
}
```

一是要有输入源（上文中是reader读入的数据）通过处理构成的模型actor、二是要有相机（camera）、三要有用于展示的窗口（window）



文献见reference

