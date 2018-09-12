/** ********************************************************************************
 *  @brief PD2: 
 *         Neste projeto em questão foi desenvolvido um algoritmo que abre um arquivo de imagem ou vídeo e permite ao usuário 
 *         clicar sobre um ponto na área da imagem ou vídeo, mostrando os valores da intensidade e a posição do pixel clicado. 
 *         Além disso, também foi feita uma rotina de seleção de pixels para pintar de vermelho os pixels ``parecidos'' com o clicado, 
 *         seguindo um critério de comparação definido pela função pinta.
 *         
 *
 * @author RAPHAEL SOARES - MAT. 14/0160299
 * @version final
 * @since 26/08/2018
 *
 ********************************************************************************/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <vector>
#include <string>
#include <algorithm>    // std::transform
#include <functional>   // std::plus

using namespace cv;
using namespace std;

#define COLOR_ROWS 80
#define COLOR_COLS 630

struct Imagem {
	Mat img;
    int x2, y2;
    bool clique = false;
    bool cinza = false;
};

void help(char **argv);
void onMouse(int event, int x, int y, int flags, void* userdata);
void imagem(char** argv);

int main( int argc, char** argv ) {
    if (argc < 2) {
        help(argv);
        return 0;
    }

    if(*argv[1] == '1' || *argv[1] == '3') 
        imagem(argv);
    else help(argv);

    return 0;
}

void help(char** argv ) {
    cout << "\n"
         << "Digite 2 e <path/imagem> para Req. 2 ou 3 e <path/video> para Req. 3 ou 4 para Req. 4\n"
         << "Por exemplo: \n" << argv[0] <<" 2 fruits.jpg\n" << endl;
}

void onMouse(int event, int x, int y, int flags, void* userdata) {
    if ( event == EVENT_LBUTTONDOWN ) {
        Imagem* img = (Imagem*) userdata;
        if(img->clique == true) {
            line(img->img, Point(img->x2,img->y2), Point(x, y), Scalar(0,0,255), 4);
            img->clique = false;
            printf("Comprimento: %.3f\n", sqrt((img->x2 - x)*(img->x2 - x) + (img->y2 - y)*(img->y2 - y)));
        } else {
            img->clique = true;
        }
        img->x2 = x;
        img->y2 = y;
        Vec3b intensity = img->img.at< Vec3b >(y, x);
        uchar blue_clique  = intensity[0];
        uchar green_clique = intensity[1];
        uchar red_clique   = intensity[2];

        float luminance = 1 - (0.299*(int)red_clique + 0.587*(int)green_clique + 0.114*(int)blue_clique) / 255;
        Scalar textColor;
        luminance < 0.5 ? textColor = Scalar(0,0,0) : textColor = Scalar(255,255,255);
        Mat colorArray; /* Mat para ser usado na "caixa" color que auxilia na verificação do progr. */
        colorArray = Mat(COLOR_ROWS, COLOR_COLS, CV_8UC3, Scalar(blue_clique,green_clique,red_clique));


        string rgbText = "(" + to_string(x) + "," + to_string(y) +
                ") " + "=" + " [R = " + to_string(red_clique) + ", G = " + 
                to_string(green_clique) + ", B = " + to_string(blue_clique) + "]";

        putText(colorArray, rgbText, Point2d(20, COLOR_ROWS - 20), FONT_HERSHEY_SIMPLEX, 0.8, textColor);
        imshow("Color", colorArray);

        cout << "Em (x,y) = (" << x << ", " << y 
             << "), onde y = coluna e x = linha, temos: " << endl 
             << "(blue, green, red) = (" << (uint)  blue_clique 
             << ", " << (uint)green_clique << ", " << (uint)  red_clique << ")" << endl;
    
        // pinta(&img, intensity);
    } 
}

void imagem (char **argv) {
    Imagem imagem;
    imagem.img = imread(argv[2], -1);

    if ( imagem.img.empty() ) {
        cout << "Imagem não encontrada!" << endl;
        return;
    }

    namedWindow("Req1", WINDOW_AUTOSIZE);
    Mat clone = imagem.img.clone();
    // cout << "Teste" << clone.at<Vec3b>(Point(0,0));
    for(;;) {
        setMouseCallback("Req1", onMouse, &imagem);
        imshow("Req1", imagem.img);
        if((char)waitKey(33) == 27) break;
        if((int)waitKey(33) == 'r') imagem.img = clone.clone();
    }
}