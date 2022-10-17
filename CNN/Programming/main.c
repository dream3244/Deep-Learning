#include <stdio.h>
#include <stdlib.h>
#include "cv.h" //把头文件包括进来
BMPImage* Convolution2D(BMPImage* img,double const kernel[3][3]);
int main(int argc, char* argv[])
{
    BMPImage* img = LoadBMPImage("girl.bmp");
    BMPImage* convdImg1;
    BMPImage* convdImg2;
    BMPImage* convdImg3;
    /*上面都是老师给的，现在下面就是我的代码编写区*/
    //高斯核（高斯模糊）
    double guassian[3][3] = {{0.089 , 0.112 , 0.089},
                            {0.112 , 0.100 , 0.112},
                            {0.089 , 0.112 , 0.089}};
    //边缘检测核
    double outline[3][3] = {{-1./8 , -1./8 , -1./8},
                            {-1./8. , 1, -1./8},
                             {-1./8 , -1./8 , -1./8}};
    //浮雕核
    double sculpture[3][3] = {{-1./2, -0 , -0},
                                {-0. , 0, -0},
                                {0 , 0 , 1./2}};
    convdImg1 = Convolution2D(img,outline);
    convdImg2 = Convolution2D(img,guassian);
    convdImg3 = Convolution2D(img,sculpture);
    /*以上三行执行完保存操作了，下面进行判断成不成功*/
    if(SaveBMPImage("outline.bmp", convdImg1))
    {
        printf("outline.bmp image processed.\n");
    }
    else
    {
        printf("fault");
    }
    if(SaveBMPImage("guassian.bmp", convdImg2))
    {
        printf("guassian.bmp image processed.\n");
    }
    else
    {
        printf("fault");
    }
    if(SaveBMPImage("sculpture.bmp", convdImg3))
    {
        printf("sculpture.bmp image processed.\n");
    }
    else
    {
        printf("fault");
    }
    //现在把指针释放掉
    free(convdImg1);
    free(convdImg2);
    free(convdImg3);
    return 0;
}
//重点来了：子函数，计算卷积，返回指向图片的指针
BMPImage* Convolution2D(BMPImage* img,double const kernel[3][3])
{
    BMPImage* newImg = NULL;
    long a = img->height;//位图高度
    long b = img->width;//位图宽度
    newImg = (BMPImage* )malloc(sizeof(BMPImage));
    newImg->width = img->width;
    newImg->channels = img->channels;
    newImg->height = img->height;
    newImg->imageData = (PixelBlock*)malloc(sizeof(PixelBlock)*(img->height)*(img->width));
    double sum=0; //每一个位点卷积求和加到sum上，完毕之后把sum赋给newImg的同个位点的数据。
    for(int i=0;i<a;i++)//大循环控制每一行
    {
        for(int j=0;j<b;j++)//小循环控制每一列
        //遍历所有的像素点
        {
            sum = 0;  //先赋初值
            //第一步，考虑图片四个角
            if(i==0&&j==0)//第一行第一个像素点，上边界与左边界不用加
            {
                sum = 0;
                sum = img->imageData[i*b+j].B*kernel[1][1]+
                    img->imageData[i*b+j+1].B*kernel[1][2]+
                    img->imageData[(i+1)*b+j].B*kernel[2][1]+
                    img->imageData[(i+1)*b+j+1].B*kernel[2][2];
                newImg->imageData[i*b+j].B = (unsigned char)sum;  //强制类型转换：为了编译不要warnings
                    //这是B方向的，接下来重复在R和G颜色上
                sum = 0;
                sum = img->imageData[i*b+j].R*kernel[1][1]+
                    img->imageData[i*b+j+1].R*kernel[1][2]+
                    img->imageData[(i+1)*b+j].R*kernel[2][1]+
                    img->imageData[(i+1)*b+j+1].R*kernel[2][2];
                newImg->imageData[i*b+j].R = (unsigned char)sum;
                sum = 0;
                sum = img->imageData[i*b+j].G*kernel[1][1]+
                    img->imageData[i*b+j+1].G*kernel[1][2]+
                    img->imageData[(i+1)*b+j].G*kernel[2][1]+
                    img->imageData[(i+1)*b+j+1].G*kernel[2][2];
                newImg->imageData[i*b+j].G = (unsigned char)sum;
            }
            else if(i==0&&j==b-1)//第一行最后一个像素点
            {
                sum = 0;
                sum = img->imageData[i*b+j-1].B*kernel[1][0]+
                    img->imageData[i*b+j].B*kernel[1][1]+
                    img->imageData[(i+1)*b+j-1].B*kernel[2][0]+
                    img->imageData[(i+1)*b+j].B*kernel[2][1];
                newImg->imageData[i*b+j].B = (unsigned char)sum;
                sum = 0;
                sum = img->imageData[i*b+j-1].R*kernel[1][0]+
                    img->imageData[i*b+j].R*kernel[1][1]+
                    img->imageData[(i+1)*b+j-1].R*kernel[2][0]+
                    img->imageData[(i+1)*b+j].R*kernel[2][1];
                newImg->imageData[i*b+j].R = (unsigned char)sum;
                sum = 0;
                sum = img->imageData[i*b+j-1].G*kernel[1][0]+
                    img->imageData[i*b+j].G*kernel[1][1]+
                    img->imageData[(i+1)*b+j-1].G*kernel[2][0]+
                    img->imageData[(i+1)*b+j].G*kernel[2][1];
                newImg->imageData[i*b+j].G = (unsigned char)sum;
            }
            else if(i==a-1&&j==0)//最后一行第一个像素点
            {
                sum = 0;
                sum = img->imageData[(i-1)*b+j].B*kernel[0][1]+
                    img->imageData[(i-1)*b+j+1].B*kernel[0][2]+
                    img->imageData[i*b+j].B*kernel[1][1]+
                    img->imageData[i*b+j+1].B*kernel[1][2];
                newImg->imageData[i*b+j].B = (unsigned char)sum;
                sum = 0;
                sum = img->imageData[(i-1)*b+j].R*kernel[0][1]+
                    img->imageData[(i-1)*b+j+1].R*kernel[0][2]+
                    img->imageData[i*b+j].R*kernel[1][1]+
                    img->imageData[i*b+j+1].R*kernel[1][2];
                newImg->imageData[i*b+j].R = (unsigned char)sum;
                sum = 0;
                sum = img->imageData[(i-1)*b+j].G*kernel[0][1]+
                    img->imageData[(i-1)*b+j+1].G*kernel[0][2]+
                    img->imageData[i*b+j].G*kernel[1][1]+
                    img->imageData[i*b+j+1].G*kernel[1][2];
                newImg->imageData[i*b+j].G = (unsigned char)sum;
            }
            else if(i==a-1&&j==b-1)//最后一行最后一个像素点
            {
                sum = 0;
                sum = img->imageData[(i-1)*b+j-1].B*kernel[0][0]+
                    img->imageData[(i-1)*b+j].B*kernel[0][1]+
                    img->imageData[i*b+j-1].B*kernel[1][0]+
                    img->imageData[i*b+j].B*kernel[1][1];
                newImg->imageData[i*b+j].B = (unsigned char)sum;
                sum = 0;
                sum = img->imageData[(i-1)*b+j-1].R*kernel[0][0]+
                    img->imageData[(i-1)*b+j].R*kernel[0][1]+
                    img->imageData[i*b+j-1].R*kernel[1][0]+
                    img->imageData[i*b+j].R*kernel[1][1];
                newImg->imageData[i*b+j].R = (unsigned char)sum;
                sum = 0;
                sum = img->imageData[(i-1)*b+j-1].G*kernel[0][0]+
                    img->imageData[(i-1)*b+j].G*kernel[0][1]+
                    img->imageData[i*b+j-1].G*kernel[1][0]+
                    img->imageData[i*b+j].G*kernel[1][1];
                newImg->imageData[i*b+j].G = (unsigned char)sum;
            }
            //第二步，考虑第一行,上边界为0
            else if(i==0)
            {
                sum = 0;
                sum = img->imageData[i*b+j-1].B*kernel[1][0]+
                    img->imageData[i*b+j].B*kernel[1][1]+
                    img->imageData[i*b+j+1].B*kernel[1][2]+
                    img->imageData[(i+1)*b+j-1].B*kernel[2][0]+
                    img->imageData[(i+1)*b+j].B*kernel[2][1]+
                    img->imageData[(i+1)*b+j+1].B*kernel[2][2];
                newImg->imageData[i*b+j].B = (unsigned char)sum;
                sum = 0;
                sum = img->imageData[i*b+j-1].R*kernel[1][0]+
                    img->imageData[i*b+j].R*kernel[1][1]+
                    img->imageData[i*b+j+1].R*kernel[1][2]+
                    img->imageData[(i+1)*b+j-1].R*kernel[2][0]+
                    img->imageData[(i+1)*b+j].R*kernel[2][1]+
                    img->imageData[(i+1)*b+j+1].R*kernel[2][2];
                newImg->imageData[i*b+j].R = (unsigned char)sum;
                sum = 0;
                sum = img->imageData[i*b+j-1].G*kernel[1][0]+
                    img->imageData[i*b+j].G*kernel[1][1]+
                    img->imageData[i*b+j+1].G*kernel[1][2]+
                    img->imageData[(i+1)*b+j-1].G*kernel[2][0]+
                    img->imageData[(i+1)*b+j].G*kernel[2][1]+
                    img->imageData[(i+1)*b+j+1].G*kernel[2][2];
                newImg->imageData[i*b+j].G = (unsigned char)sum;
            }
            //第三步，考虑最后一行，下边界为0
            else if(i==a-1)
            {
                sum = 0;
                sum = img->imageData[(i-1)*b+j-1].B*kernel[0][0]+
                    img->imageData[(i-1)*b+j].B*kernel[0][1]+
                    img->imageData[(i-1)*b+j+1].B*kernel[0][2]+
                    img->imageData[i*b+j-1].B*kernel[1][0]+
                    img->imageData[i*b+j].B*kernel[1][1]+
                    img->imageData[i*b+j+1].B*kernel[1][2];
                newImg->imageData[i*b+j].B = (unsigned char)sum;
                sum = 0;
                sum = img->imageData[(i-1)*b+j-1].R*kernel[0][0]+
                    img->imageData[(i-1)*b+j].R*kernel[0][1]+
                    img->imageData[(i-1)*b+j+1].R*kernel[0][2]+
                    img->imageData[i*b+j-1].R*kernel[1][0]+
                    img->imageData[i*b+j].R*kernel[1][1]+
                    img->imageData[i*b+j+1].R*kernel[1][2];
                newImg->imageData[i*b+j].R = (unsigned char)sum;
                sum = 0;
                sum = img->imageData[(i-1)*b+j-1].G*kernel[0][0]+
                    img->imageData[(i-1)*b+j].G*kernel[0][1]+
                    img->imageData[(i-1)*b+j+1].G*kernel[0][2]+
                    img->imageData[i*b+j-1].G*kernel[1][0]+
                    img->imageData[i*b+j].G*kernel[1][1]+
                    img->imageData[i*b+j+1].G*kernel[1][2];
                newImg->imageData[i*b+j].G = (unsigned char)sum;
            }
            //考虑第一列，左边界为0
            else if(j==0)
            {
                sum = 0;
                sum = img->imageData[(i-1)*b+j].B*kernel[0][1]+
                    img->imageData[(i-1)*b+j+1].B*kernel[0][2]+
                    img->imageData[i*b+j].B*kernel[1][1]+
                    img->imageData[i*b+j+1].B*kernel[1][2]+
                    img->imageData[(i+1)*b+j].B*kernel[2][1]+
                    img->imageData[(i+1)*b+j+1].B*kernel[2][2];
                newImg->imageData[i*b+j].B = (unsigned char)sum;
                sum = 0;
                sum = img->imageData[(i-1)*b+j].R*kernel[0][1]+
                    img->imageData[(i-1)*b+j+1].R*kernel[0][2]+
                    img->imageData[i*b+j].R*kernel[1][1]+
                    img->imageData[i*b+j+1].R*kernel[1][2]+
                    img->imageData[(i+1)*b+j].R*kernel[2][1]+
                    img->imageData[(i+1)*b+j+1].R*kernel[2][2];
                newImg->imageData[i*b+j].R = (unsigned char)sum;
                sum = 0;
                sum = img->imageData[(i-1)*b+j].G*kernel[0][1]+
                    img->imageData[(i-1)*b+j+1].G*kernel[0][2]+
                    img->imageData[i*b+j].G*kernel[1][1]+
                    img->imageData[i*b+j+1].G*kernel[1][2]+
                    img->imageData[(i+1)*b+j].G*kernel[2][1]+
                    img->imageData[(i+1)*b+j+1].G*kernel[2][2];
                newImg->imageData[i*b+j].G = (unsigned char)sum;
            }
            //考虑最后一列，右边界为0
            else if(j==b-1)
            {
                sum = 0;
                sum = img->imageData[(i-1)*b+j-1].B*kernel[0][0]+
                    img->imageData[(i-1)*b+j].B*kernel[0][1]+
                    img->imageData[i*b+j-1].B*kernel[1][0]+
                    img->imageData[i*b+j].B*kernel[1][1]+
                    img->imageData[(i+1)*b+j-1].B*kernel[2][0]+
                    img->imageData[(i+1)*b+j].B*kernel[2][1];
                newImg->imageData[i*b+j].B = (unsigned char)sum;
                sum = 0;
                sum = img->imageData[(i-1)*b+j-1].R*kernel[0][0]+
                    img->imageData[(i-1)*b+j].R*kernel[0][1]+
                    img->imageData[i*b+j-1].R*kernel[1][0]+
                    img->imageData[i*b+j].R*kernel[1][1]+
                    img->imageData[(i+1)*b+j-1].R*kernel[2][0]+
                    img->imageData[(i+1)*b+j].R*kernel[2][1];
                newImg->imageData[i*b+j].R = (unsigned char)sum;
                sum = 0;
                sum = img->imageData[(i-1)*b+j-1].G*kernel[0][0]+
                    img->imageData[(i-1)*b+j].G*kernel[0][1]+
                    img->imageData[i*b+j-1].G*kernel[1][0]+
                    img->imageData[i*b+j].G*kernel[1][1]+
                    img->imageData[(i+1)*b+j-1].G*kernel[2][0]+
                    img->imageData[(i+1)*b+j].G*kernel[2][1];
                newImg->imageData[i*b+j].G = (unsigned char)sum;
            }
            //正常情况
            else
            {
                sum = 0;
                sum = img->imageData[(i-1)*b+j-1].B*kernel[0][0]+
                    img->imageData[(i-1)*b+j].B*kernel[0][1]+
                    img->imageData[(i-1)*b+j+1].B*kernel[0][2]+
                    img->imageData[i*b+j-1].B*kernel[1][0]+
                    img->imageData[i*b+j].B*kernel[1][1]+
                    img->imageData[i*b+j+1].B*kernel[1][2]+
                    img->imageData[(i+1)*b+j-1].B*kernel[2][0]+
                    img->imageData[(i+1)*b+j].B*kernel[2][1]+
                    img->imageData[(i+1)*b+j+1].B*kernel[2][2];
                newImg->imageData[i*b+j].B = (unsigned char)sum;
                sum = 0;
                sum = img->imageData[(i-1)*b+j-1].R*kernel[0][0]+
                    img->imageData[(i-1)*b+j].R*kernel[0][1]+
                    img->imageData[(i-1)*b+j+1].R*kernel[0][2]+
                    img->imageData[i*b+j-1].R*kernel[1][0]+
                    img->imageData[i*b+j].R*kernel[1][1]+
                    img->imageData[i*b+j+1].R*kernel[1][2]+
                    img->imageData[(i+1)*b+j-1].R*kernel[2][0]+
                    img->imageData[(i+1)*b+j].R*kernel[2][1]+
                    img->imageData[(i+1)*b+j+1].R*kernel[2][2];
                newImg->imageData[i*b+j].R = (unsigned char)sum;
                sum = 0;
                sum = img->imageData[(i-1)*b+j-1].G*kernel[0][0]+
                    img->imageData[(i-1)*b+j].G*kernel[0][1]+
                    img->imageData[(i-1)*b+j+1].G*kernel[0][2]+
                    img->imageData[i*b+j-1].G*kernel[1][0]+
                    img->imageData[i*b+j].G*kernel[1][1]+
                    img->imageData[i*b+j+1].G*kernel[1][2]+
                    img->imageData[(i+1)*b+j-1].G*kernel[2][0]+
                    img->imageData[(i+1)*b+j].G*kernel[2][1]+
                    img->imageData[(i+1)*b+j+1].G*kernel[2][2];
                newImg->imageData[i*b+j].G = (unsigned char)sum;
            }
        }
    }
    return newImg;
}



