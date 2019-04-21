## Especificações da máquina:

Dell Inspiron I14-5458-D37P

PROCESSADOR: Intel Core i5 5200U de 2.2 até 2.7 GHz; <br>
MEMÓRIA RAM: 8 GB DDR3L 1600 MHz; <br>
PLACA DE VÍDEO: Intel HD Graphics 5500; <br>
DISCO RÍGIDO: 1 TB; <br>
SO: Ubuntu 16.04 LTS

## Descrição da solução

A solução para o problema de classificação foi implementada toda em Python utilizando-se principalmente da biblioteca tensorflow e de sua API, Keras.

O modelo foi construido a partir de uma rede neural convolucional com a seguinte arquitetura:

* **Camada de entrada**
* **Camada convolucional #1**
  * Função de ativação Relu
  * 32 filtros
  * Kernel size 3 por 3
* **Camada de max pooling #1**
  * Pool size 2 por 2
* **Camada convolucional #2**
  * Função de ativação Relu
  * 64 filtros
  * Kernel 3 por 3
* **Camada de max pooling #2**
  * Pool size 2 por 2
* **Camada convolucional #3**
  * Função de ativação Relu
  * 128 filtros
  * Kernel 3 por 3
* **Camada de max pooling #3**
  * Pool size 2 por 2
* **Camada densa**
  * Regularizador L2
  * Taxa de dropout de 50%
* **Camada de saída**
  * Regularizador L2
  * Função de ativação softmax

## Packages

Os *python packages* utilizados para **filtrar o dataset**, **executar o módulo de preprocessamento** de imagens e **treinar a rede**, juntamente com suas respectivas versões estão no arquivo *requirements.txt*. É possível instalar todos os packages através do comando:

$ pip install -r requirements.txt

## Treinando a rede

É possível re-executar todos os passos do treinamento e salvar um novo modelo a partir da execução do arquivo *ModelNotebook.ipynb* ou *detector.py*.

Nota: Na última célula do arquivo *ModelNotebook.ipynb* há o trecho de código que salva os parâmetros do modelo treinado, se você quer usar o modelo original apresentado neste repositório, não execute esta célula.

### Exemplo de validação

Como apenas 1203 do total de imagens do dataset tem um rótulo, todas estas imagens foram utilizadas pra treinamento e validação do modelo. Todavia, 20 imagens foram extraídas do dataset completo e rotuladas manualmente para testar rapidamente a performance de classificação do modelo após o treinamento e validação.

O trecho do código responsável por aferir o desempenho do modelo sob estes dados está na seção **Predictions with evaluation_faces dataset**, logo após a seção **Results** no arquivo *ModelNotebook.ipynb*. 

Vale ressaltar que as 10 primeiras imagens deste dataset são de pessoas sorrindo e as 10 últimas são de pessoas que não estão sorrindo.

## Executando na sua máquina

Um script de um exemplo de execução do modelo está contido no arquivo **Testing.ipynb**. Para executá-lo, é necessário apenas:

* Instalar a biblioteca tensorflow 1.13.1
* Carregar o módulo transform_module.py
* Carregar o arquivo *model*, que contém os parâmetros da rede neural treinada

Como teste, foram utilizadas 6 imagens próprias, carregadas no próprio Notebook. 