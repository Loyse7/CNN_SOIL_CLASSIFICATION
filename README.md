# Classificação de Imagens de Satélite com CNN

## Identificação

- **Nome da estudante:** Loyse Kelly Cardoso Diniz  
- **Disciplina:** Aprendizado Profundo  
- **Modelo implementado:** CNN (Convolutional Neural Network)  
- **Dataset utilizado:** EuroSAT RGB  
- **Link do dataset:** https://huggingface.co/datasets/blanchon/EuroSAT_RGB  

---

## Descrição do problema

A classificação automática de imagens de satélite é uma tarefa importante em aplicações de sensoriamento remoto, monitoramento ambiental, agricultura de precisão e planejamento urbano. Identificar corretamente diferentes tipos de cobertura e uso do solo permite gerar informações estratégicas para tomada de decisão em diversas áreas.

Este projeto aborda o uso de Redes Neurais Convolucionais (CNN) para automatizar a classificação de imagens RGB de satélite, realizando a categorização entre diferentes classes de uso e cobertura do solo. O modelo atua como um sistema inteligente capaz de aprender padrões espaciais complexos, como vegetação, corpos d’água, áreas urbanas e zonas industriais.

---

## Dataset

O conjunto de dados utilizado é o **EuroSAT RGB**, amplamente utilizado em tarefas de visão computacional aplicada a sensoriamento remoto. O dataset contém imagens de satélite obtidas do programa Sentinel-2.

- **Número de classes:** 10 classes mutuamente exclusivas  
- **Total de imagens utilizadas:** 16.200  
- **Resolução:** 64x64 pixels RGB  

### Classes

- AnnualCrop  
- Forest  
- HerbaceousVegetation  
- Highway  
- Industrial  
- Pasture  
- PermanentCrop  
- Residential  
- River  
- SeaLake  

### Divisão dos dados

Os dados foram divididos de maneira estratificada utilizando semente aleatória fixa:

- **Treinamento (70%) — 11.340 imagens**
- **Validação (15%) — 2.430 imagens**
- **Teste (15%) — 2.430 imagens**

---

## Data Augmentation

Durante o treinamento, foram aplicadas transformações para aumentar a variabilidade dos dados e reduzir overfitting:

```python
transformacao_treino = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor()
])
```

Essas transformações ajudam a CNN a aprender padrões mais robustos.

---

## Modelo

Foi construída uma arquitetura convolucional profunda (**CNN customizada**) em PyTorch.

A rede possui **4 blocos convolucionais**, responsáveis por extrair características hierárquicas das imagens.

As primeiras camadas aprendem:
- bordas  
- texturas  
- padrões locais  

As camadas profundas aprendem:
- estruturas urbanas  
- padrões agrícolas  
- rios e lagos  
- áreas industriais  

A etapa classificadora recebe as features extraídas e realiza a predição final entre as 10 classes.

### Arquitetura da rede

```text
CNN_EuroSAT(
  Bloco Conv 1: Conv2D(3→32) + BatchNorm + ReLU + Conv2D + MaxPool + Dropout
  Bloco Conv 2: Conv2D(32→64) + BatchNorm + ReLU + Conv2D + MaxPool + Dropout
  Bloco Conv 3: Conv2D(64→128) + BatchNorm + ReLU + Conv2D + MaxPool + Dropout
  Bloco Conv 4: Conv2D(128→256) + BatchNorm + ReLU + Conv2D + MaxPool + Dropout

  Classificador:
  Flatten
  Linear(4096 → 512)
  ReLU
  Dropout(0.5)
  Linear(512 → 256)
  ReLU
  Dropout(0.4)
  Linear(256 → 10)
)
```

### Total de parâmetros

- **3.407.274 parâmetros treináveis**

---

## Ambiente

- **Linguagem:** Python  
- **Bibliotecas:** torch, torchvision, numpy, matplotlib, scikit-learn, datasets  
- **Versão do Python:** 3.12+  
- **Uso de GPU:** Sim  

---

## Principais resultados

Métricas obtidas no conjunto de teste:

- **Acurácia Global:** **96,30%**
- **Precisão (Macro):** **96,21%**
- **Recall (Macro):** **96,26%**
- **F1-Score (Macro):** **96,20%**

O modelo apresentou excelente capacidade de generalização, classificando corretamente aproximadamente **96 em cada 100 imagens**.

### Classes com melhor desempenho

- Forest → 99,09% F1  
- Industrial → 99,15% F1  
- Residential → 99,22% F1  
- SeaLake → 99,12% F1  

### Classes mais desafiadoras

- PermanentCrop → 91,06% F1  
- River → 94,12% F1  
- HerbaceousVegetation → 92,99% F1  

As principais confusões ocorreram entre classes com padrões visuais semelhantes, especialmente categorias relacionadas à vegetação.

---

## Técnicas utilizadas

Durante o desenvolvimento foram aplicadas técnicas para melhorar o desempenho do modelo:

- Data Augmentation  
- Batch Normalization  
- Dropout  
- Early Stopping  
- AdamW Optimizer  
- Matriz de Confusão  
- Visualização de Feature Maps  

Essas técnicas ajudaram a melhorar:
- generalização  
- estabilidade do treinamento  
- redução de overfitting  

---

## Como executar

```bash
# Ativar GPU (Google Colab)
Ambiente de execução > Alterar tipo de ambiente > GPU T4

# Instalar dependências
pip install torch torchvision datasets matplotlib scikit-learn

# Executar notebook
CNN_EuroSAT.ipynb
```

---

## Conclusão

Este projeto demonstrou a eficácia de Redes Neurais Convolucionais na tarefa de classificação de imagens de satélite. A arquitetura proposta foi capaz de extrair características espaciais relevantes e distinguir com alta precisão diferentes tipos de uso e cobertura do solo.

Os resultados obtidos mostram que CNNs são ferramentas poderosas para aplicações em sensoriamento remoto, monitoramento ambiental e análise geoespacial.
