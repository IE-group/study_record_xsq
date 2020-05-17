### Study Report 



**VTK三维重建方法总结**

链接：

https://blog.csdn.net/Q1302182594/article/details/45892995?depth_1-utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-8&utm_source=distribute.pc_relevant.none-task-blog-BlogCommendFromBaidu-8



以下是VTK读取序列的DICOM医学医学图片，用Marchingcube进行重建，并保存为obj文件

```vtk

#include "vtkRenderer.h"

#include "vtkRenderWindow.h"

#include "vtkRenderWindowInteractor.h"

#include "vtkDICOMImageReader.h"

#include "vtkMarchingCubes.h"

#include "vtkStripper.h"

#include "vtkPolyDataMapper.h"

#include "vtkActor.h"

#include "vtkSmartPointer.h"

#include "vtkProperty.h"

#include "vtkCamera.h"

#include "vtkOutlineFilter.h"

#include "vtkOBJExporter.h"

 

void main()

{

	vtkSmartPointer<vtkRenderer>ren=vtkSmartPointer<vtkRenderer>::New();             //设置绘制者(绘制对象指针)

	vtkSmartPointer<vtkRenderWindow>renWin=vtkSmartPointer<vtkRenderWindow>::New();         //设置绘制窗口

	renWin->AddRenderer(ren);                                 //将绘制者加入绘制窗口

 

	vtkSmartPointer<vtkRenderWindowInteractor>iren=vtkSmartPointer<vtkRenderWindowInteractor>::New();//设置绘制交互操作窗口的

	iren->SetRenderWindow(renWin);                             //将绘制窗口添加到交互窗口

 

	vtkSmartPointer<vtkDICOMImageReader>Reader=vtkSmartPointer<vtkDICOMImageReader>::New(); //创建读取dicom图片指针对象

	Reader->SetDirectoryName("E:\\dcm\\dcm"); //设置医学图像文件夹路径

	Reader->SetDataByteOrderToLittleEndian();

	Reader->Update();

	cout<<"读取数据完成"<<endl;

 

 

	//////////////////////////////////////////////

	vtkSmartPointer<vtkMarchingCubes>marchingcube=vtkSmartPointer<vtkMarchingCubes>::New();       //建立一个Marching Cubes 算法的对象	

	//marchingcube->SetInput((vtkDataSet *)Reader->GetOutput());      //获得所读取的数据

	marchingcube->SetInputConnection(Reader->GetOutputPort());      //第二种读取数据的方法

	marchingcube->SetValue(0,140);                                  //提取出灰度值为45的东西

 

	vtkSmartPointer<vtkStripper>Stripper=vtkSmartPointer<vtkStripper>::New();                   //建立三角带对象

	Stripper->SetInput( marchingcube->GetOutput() );              //将生成的三角片连接成三角带

 

	vtkSmartPointer<vtkPolyDataMapper>Mapper=vtkSmartPointer<vtkPolyDataMapper>::New();   //建立一个数据映射对象

	Mapper->SetInput(Stripper->GetOutput() );            //将三角带映射为几何数据

	Mapper->ScalarVisibilityOff();

 

	vtkSmartPointer<vtkActor>actor=vtkSmartPointer<vtkActor>::New();                            //建立一个代表皮肤的演员

	actor->SetMapper(Mapper);                                  //获得皮肤几何数据的属性

	actor->GetProperty()->SetDiffuseColor(1, .49, .25);            //设置皮肤颜色的属性//(1, .49, .25)

	//actor->GetProperty()->SetDiffuseColor(1, 1, .25);

	actor->GetProperty()->SetSpecular(0.3);                         //设置反射率

	actor->GetProperty()->SetSpecularPower(20);                    //设置反射光强度 

	actor->GetProperty()->SetOpacity(1.0);                 

	actor->GetProperty()->SetColor(1,0,0);                 //设置角色的颜色

	actor->GetProperty()->SetRepresentationToWireframe();

 

 

	vtkSmartPointer<vtkCamera>aCamera=vtkSmartPointer<vtkCamera>::New();     // 创建摄像机

	aCamera->SetViewUp ( 0, 0, -1 );        //设置相机的“上”方向

	aCamera->SetPosition ( 0, 1, 0 );       //位置：世界坐标系，设置相机位置

	aCamera->SetFocalPoint( 0, 0, 0 );     //焦点：世界坐标系，控制相机方向

	aCamera->ComputeViewPlaneNormal();     //重置视平面法向，基于当前的位置和焦点。否则会出现斜推剪切效果

 

	///////////////////////////////////////

	vtkSmartPointer<vtkOutlineFilter>outlinefilter=vtkSmartPointer<vtkOutlineFilter>::New();

	outlinefilter->SetInputConnection(Reader->GetOutputPort());

 

	vtkSmartPointer<vtkPolyDataMapper>outlineMapper=vtkSmartPointer<vtkPolyDataMapper>::New();

	outlineMapper->SetInputConnection(outlinefilter->GetOutputPort());

 

	vtkSmartPointer<vtkActor>OutlineActor=vtkSmartPointer<vtkActor>::New();

	OutlineActor->SetMapper(outlineMapper);

	OutlineActor->GetProperty()->SetColor(0,0,0);

	// 告诉绘制者将要在绘制窗口中进行显示的演员

	ren->AddActor(actor);

	ren->AddActor(OutlineActor);

	

	ren->SetActiveCamera(aCamera);               ////设置渲染器的相机

	ren->ResetCamera();

	aCamera->Dolly(1.5);                //使用Dolly()方法沿着视平面法向移动相机，实现放大或缩小可见角色物体

	ren->SetBackground(1,1,1);               //设置背景颜色

	//ren->ResetCameraClippingRange();

	renWin->SetSize(1000, 600);

	renWin->Render();

	iren->Initialize();

	iren->Start(); 

	vtkSmartPointer<vtkOBJExporter> porter=vtkSmartPointer<vtkOBJExporter>::New();

	porter->SetFilePrefix("E:\\PolyDataWriter.obj");

	porter->SetInput(renWin);

	porter->Write();

 

}

```



关于3D深度学习在CT影像预测早期肿瘤浸润方面超过影像专家的论文：

http://cancerres.aacrjournals.org/content/early/2018/10/02/0008-5472.CAN-18-0696