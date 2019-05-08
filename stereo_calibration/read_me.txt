Nome: Raphael Soares Ramos
Matrícula: 140160299

Este projeto foi desenvolvido foi desenvolvido em Linux 18.04 64 bits, utilizando a versão 3.2.0 do OpenCV e C++11 com compilador GCC 7.3.0.

Para compilar o programa apenas execute o comando make a partir do terminal. Em seguida:
- Use a regra r1 para gerar os mapas do requisito 1 das imagens aloe e baby usando os algoritmos sgbm e bm, conforme explica o relatório.
- Use a regra r2 parar gerar os mapas do requisito 2 e estimar as medidas do requisito 3.
- Use a regra b (bônus) para gerar os mesmos mapas do morpheus, porém para a imagem warrior retirada de https://www.cse.wustl.edu/~furukawa/research/mview/index.html.

Os dois primeiros parâmetros da r1 representam os caminhos das imagens; o terceiro representa sim ou não para executar req1; o quarto a mesma coisa só que pro req2; o quinto representa o algoritmo que dever ser utilizado (sgbm ou bm); o quinto representa se a imagem em questão é o baby ou aloe; e o último representa o tamanho w da janela.

Quando as imagens forem sendo exibidas na tela, aperte ESC para fechá-las e continuar o programa.
O último número/parâmetro nestas regras referem-se ao tamanho da janela W, tanto para o algoritmo SGBM quanto para o BM. Recomenda-se uma janela de tamanho 5 para o SGBM (este algoritmo é mais robusto e a janela é menor mesmo) e 15 para o BM.

* Importante
Para calcular as dimensões da caixa, a partir da janela Req3 que será aberta ao executar a regra r2 do make, é necessário seguir os seguintes passos:
- Primeiro meça a largura partindo do canto superior esquerdo do sofá (primeiro clique) até o canto superior direito do sofá (segundo clique);
- Depois meça a altura partindo do mesmo ponto (terceiro clique) até o canto inferior esquerdo do sofá (quarto clique);
- Por fim a profundidade deve ser obtida partir da diagonal, partindo do canto inferior direito (quinto clique) e depois o canto superior direito (sexto clique).

Caso mais do que seis cliques sejam dados, somente os 6 primeiros serão considerados e serão calculados seguindo a ordem acima.
Seguindo esses passos, creio que ao tentar reproduzir as dimensões apresentadas no relatório é possível obter boas aproximações.  

A normalização das imagens dos mapas foram realizadas seguindo a fórmula apresentada no relatório, mais precisamente: https://en.wikipedia.org/wiki/Normalization_(image_processing).
Sendo que newMax = 255 e newMin = 0, foi utilizado. 

Caso queira observar os mapas antes da normalização, basta olhar pra imagem "raw disparity" no caso da disparidade, e "filtered_disp" no caso da profundidade. Estas imagens são mostradas no final da função calcula_mapas para as imagens do requisito 1, e na função cria_mapas_morpheus para as imagens do morpheus.

O diretório data foi incluído pelo fato de que as imagens do warrior usadas no bônus não são iguais a todas as outras fornecidas.
