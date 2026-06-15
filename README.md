# Assistente Virtual de Dados — Desafio FRANQ

Assistente conversacional que transforma perguntas em linguagem natural em consultas SQL, executa-as em um banco de dados SQLite e retorna respostas claras.

Repositório: [github.com/santossribeiroo/desafio-franq-ia](https://github.com/santossribeiroo/desafio-franq-ia.git)

---

## Instruções de Execução

### 1. Clone o repositório

```bash
git clone https://github.com/santossribeiroo/desafio-franq-ia.git
cd desafio-franq-ia
```

### 2. Crie e ative o ambiente virtual

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Instale as dependências

```bash
pip install -r requirements.txt
```

### 4. Configure as variáveis de ambiente

Copie o arquivo de exemplo e preencha com sua chave:

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Abra o `.env` e adicione sua chave da API Gemini, obtida em [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey):

```env
GOOGLE_API_KEY="sua-chave-aqui"
```

### 5. Execute o projeto

```bash
python main.py
```

### 6. Rode os testes automatizados

```bash
python -m pytest tests/ -v
```

O terminal exibirá a query SQL gerada e a resposta formatada para a pergunta de exemplo.

---

## Arquitetura e Fluxo de Agentes

O pipeline é orquestrado pelo **LangGraph** como um grafo de estados com três nós sequenciais:

```
Pergunta do usuário
        │
        ▼
┌───────────────────┐
│  generate_sql     │  Lê o schema do banco e gera a query SQL
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  execute_sql      │  Executa a query no SQLite e retorna os resultados
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  format_response  │  Formata os dados em uma resposta em pt-BR
└────────┬──────────┘
         │
         ▼
  Resposta final ao usuário
```

| Nó | O que faz |
|---|---|
| `generate_sql_node` | Lê o schema real do banco em tempo de execução e instrui o modelo a retornar apenas a query SQL, sem formatação extra |
| `execute_sql_node` | Executa a query gerada. Se um nó anterior falhar, o erro é propagado sem tocar no banco |
| `format_response_node` | Usa o modelo para transformar os dados brutos em uma resposta profissional em pt-BR, ou gera uma mensagem de erro amigável quando necessário |

O modelo de linguagem utilizado é o **Gemini 2.5 Flash** via `langchain-google-genai`, com temperatura `0` para geração de SQL (máxima precisão) e `0.3` para a resposta final (tom mais natural).

---

## Exemplos de Consultas Testadas

| Pergunta | Resultado |
|---|---|
| `"Quais campanhas de marketing tiveram interação dos clientes?"` | Lista das campanhas com registros de interação |
| `"Quais são os 5 clientes que mais gastaram?"` | Ranking dos clientes por volume de gasto |

---