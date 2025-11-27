Descrição

Este projeto implementa um sistema de detecção para vídeos utilizando técnicas de visão computacional em Python. Através do arquivo main.py, o script carrega vídeos de teste (na pasta test_videos) e faz uso do módulo Detector.py para processar cada quadro, identificando e marcando objetos de interesse.
Além disso, conta com dados de modelo em model_data para suporte à inferência.

Funcionalidades

Carregamento de vídeos de entrada para detecção automática.

Utilização de classe Detector para encapsular lógica de detecção (localização de objetos, marcação de bounding boxes, etc.).

Modularização do código para facilitar extensão e manutenção.

Diretório de testes (test_videos) para validação e demonstração.

Estrutura simples para adaptação a novos modelos ou novos vídeos.

Estrutura de pastas
```text
/
├── model_data/        ← arquivos de modelo, pesos, etc.
├── test_videos/       ← vídeos de entrada para teste
├── Detector.py        ← módulo de detecção de objetos
├── main.py            ← script principal para execução
└── README.md          ← este arquivo
```
Requisitos

Python 3.x

Dependências (exemplo):

pip install opencv-python numpy


Adapte conforme os pacotes que você realmente usa (por exemplo, TensorFlow, PyTorch, etc.).

Como executar

Clone o repositório:

git clone https://github.com/VitorPio7/vis-o-computacional-a3.git
cd vis-o-computacional-a3


Verifique se os vídeos de teste estão em test_videos/.

Execute o script principal:

python main.py


O script processará cada vídeo na pasta de teste, aplicará a detecção e exibirá ou salvará os resultados (adaptar conforme a implementação).

(Opcional) Ajuste o módulo Detector.py para usar outro modelo ou outra lógica de detecção.

Personalização / Extensão

Para trocar o modelo de detecção: substitua o conteúdo em model_data/ e ajuste no Detector.py as rotinas de carregamento/inferência.

Para processar outros vídeos: adicione ao test_videos/ ou modifique main.py para apontar para outros diretórios.

Para salvar os resultados em arquivo (vídeo ou imagem): adicione lógica de gravação no main.py ou em Detector.py.

Para métricas de desempenho (FPS, tempo por quadro, etc.): modifique o código para coletar e mostrar relatórios.

Licença

Este projeto está disponível sob a licença MIT (ou outra que você escolher). Veja o arquivo LICENSE para detalhes.
