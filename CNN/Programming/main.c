#include <stdio.h>
#include <stdlib.h>
#include "cv.h" //��ͷ�ļ���������
BMPImage* Convolution2D(BMPImage* img,double const kernel[3][3]);
int main(int argc, char* argv[])
{
    BMPImage* img = LoadBMPImage("girl.bmp");
    BMPImage* convdImg1;
    BMPImage* convdImg2;
    BMPImage* convdImg3;
    /*���涼����ʦ���ģ�������������ҵĴ����д��*/
    //��˹�ˣ���˹ģ����
    double guassian[3][3] = {{0.089 , 0.112 , 0.089},
                            {0.112 , 0.100 , 0.112},
                            {0.089 , 0.112 , 0.089}};
    //��Ե����
    double outline[3][3] = {{-1./8 , -1./8 , -1./8},
                            {-1./8. , 1, -1./8},
                             {-1./8 , -1./8 , -1./8}};
    //�����
    double sculpture[3][3] = {{-1./2, -0 , -0},
                                {-0. , 0, -0},
                                {0 , 0 , 1./2}};
    convdImg1 = Convolution2D(img,outline);
    convdImg2 = Convolution2D(img,guassian);
    convdImg3 = Convolution2D(img,sculpture);
    /*��������ִ���걣������ˣ���������жϳɲ��ɹ�*/
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
    //���ڰ�ָ���ͷŵ�
    free(convdImg1);
    free(convdImg2);
    free(convdImg3);
    return 0;
}
//�ص����ˣ��Ӻ�����������������ָ��ͼƬ��ָ��
BMPImage* Convolution2D(BMPImage* img,double const kernel[3][3])
{
    BMPImage* newImg = NULL;
    long a = img->height;//λͼ�߶�
    long b = img->width;//λͼ���
    newImg = (BMPImage* )malloc(sizeof(BMPImage));
    newImg->width = img->width;
    newImg->channels = img->channels;
    newImg->height = img->height;
    newImg->imageData = (PixelBlock*)malloc(sizeof(PixelBlock)*(img->height)*(img->width));
    double sum=0; //ÿһ��λ������ͼӵ�sum�ϣ����֮���sum����newImg��ͬ��λ������ݡ�
    for(int i=0;i<a;i++)//��ѭ������ÿһ��
    {
        for(int j=0;j<b;j++)//Сѭ������ÿһ��
        //�������е����ص�
        {
            sum = 0;  //�ȸ���ֵ
            //��һ��������ͼƬ�ĸ���
            if(i==0&&j==0)//��һ�е�һ�����ص㣬�ϱ߽�����߽粻�ü�
            {
                sum = 0;
                sum = img->imageData[i*b+j].B*kernel[1][1]+
                    img->imageData[i*b+j+1].B*kernel[1][2]+
                    img->imageData[(i+1)*b+j].B*kernel[2][1]+
                    img->imageData[(i+1)*b+j+1].B*kernel[2][2];
                newImg->imageData[i*b+j].B = (unsigned char)sum;  //ǿ������ת����Ϊ�˱��벻Ҫwarnings
                    //����B����ģ��������ظ���R��G��ɫ��
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
            else if(i==0&&j==b-1)//��һ�����һ�����ص�
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
            else if(i==a-1&&j==0)//���һ�е�һ�����ص�
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
            else if(i==a-1&&j==b-1)//���һ�����һ�����ص�
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
            //�ڶ��������ǵ�һ��,�ϱ߽�Ϊ0
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
            //���������������һ�У��±߽�Ϊ0
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
            //���ǵ�һ�У���߽�Ϊ0
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
            //�������һ�У��ұ߽�Ϊ0
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
            //�������
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



