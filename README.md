# Assistente Virtual de Dados — Desafio FRANQ

Assistente conversacional que transforma perguntas em linguagem natural em consultas SQL, executa-as em um banco de dados SQLite e retorna respostas claras em Português do Brasil, com interface visual interativa via Streamlit.

Repositório: [github.com/santossribeiroo/desafio-franq-ia](https://github.com/santossribeiroo/desafio-franq-ia.git)

---

## Pré-requisitos

- **Python 3.11+** instalado — [python.org/downloads](https://www.python.org/downloads/)
- **Chave de API do Google Gemini** — obtida gratuitamente em [aistudio.google.com](https://aistudio.google.com)
- **Git** — [git-scm.com](https://git-scm.com/)

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

As seguintes bibliotecas serão instaladas automaticamente:

| Biblioteca | Uso |
|---|---|
| `langchain` + `langgraph` | Orquestração do agente como grafo de estados |
| `langchain-google-genai` | Integração com os modelos Gemini |
| `google-generativeai` | SDK oficial do Google para a API Gemini |
| `streamlit` | Interface visual interativa |
| `plotly` | Gráficos interativos |
| `pandas` | Manipulação dos resultados do banco |
| `sqlparse` | Formatação da query SQL gerada |
| `sqlalchemy` | Utilitários de banco de dados |
| `python-dotenv` | Carregamento de variáveis de ambiente |

### 4. Configure a chave da API Gemini

Crie sua chave gratuitamente em **[aistudio.google.com](https://aistudio.google.com)** → clique em **"Get API key"**.

Copie o arquivo de exemplo e preencha com sua chave:

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Abra o arquivo `.env` e cole sua chave:

```env
GOOGLE_API_KEY="sua-chave-do-google-ai-studio-aqui"
```

### 5. Execute a interface Streamlit

```bash
streamlit run app.py
```

Acesse no navegador: [http://localhost:8501](http://localhost:8501)

### 6. (Opcional) Teste o pipeline via terminal

```bash
python main.py
```

O terminal exibirá a query SQL gerada e a resposta formatada para uma pergunta de exemplo.

---

## Arquitetura e Fluxo de Agentes

O pipeline é orquestrado pelo **LangGraph** como um grafo de estados com roteamento condicional:

```
Pergunta do usuário
        │
        ▼
┌───────────────────┐
│  generate_sql     │  Valida a pergunta e gera a query SQL (1 chamada ao LLM)
└────────┬──────────┘
         │ ──► pergunta inválida → format_response (mensagem amigável)
         │ ──► erro 429 → END (evita chamada extra desnecessária)
         ▼
┌───────────────────┐
│  execute_sql      │  Valida e executa a query no SQLite
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  format_response  │  Transforma os dados em resposta profissional em pt-BR
└────────┬──────────┘
         │
         ▼
  Resposta final ao usuário
```

| Nó | O que faz |
|---|---|
| `generate_sql_node` | Lê o schema real do banco em tempo de execução, valida se a pergunta é respondível e retorna apenas a query SQL executável |
| `execute_sql_node` | Bloqueia queries não-SELECT por segurança, valida sintaxe com `EXPLAIN QUERY PLAN` e executa no SQLite |
| `format_response_node` | Transforma os dados brutos em resposta profissional em pt-BR, ou gera mensagem de erro amigável |

O modelo utilizado é o **Gemini 2.5 Flash Lite** via `langchain-google-genai`, com temperatura `0` para geração de SQL (máxima precisão) e `0.3` para a resposta final (tom mais natural).

---

## Exemplos de Consultas Testadas

| Pergunta | Tipo de análise |
|---|---|
| `"Quais são os 5 clientes que mais gastaram?"` | Ranking com ORDER BY + LIMIT |
| `"Qual é o valor médio de compra por categoria?"` | Agregação com GROUP BY + AVG |
| `"Qual canal de compra gerou mais receita?"` | GROUP BY + SUM |
| `"Qual cidade tem mais clientes?"` | Contagem por agrupamento |
| `"Quantos chamados de suporte foram resolvidos?"` | Filtro booleano |
| `"Quais campanhas de marketing tiveram mais interações?"` | Análise de campanhas |
| `"Quantos clientes interagiram com campanhas de WhatsApp em 2024?"` | Filtro multi-coluna + data |

---

## Sugestões de Melhorias

- **Memória conversacional** para perguntas de acompanhamento com contexto das interações anteriores.
- **Validação adicional da query** com análise semântica antes da execução.
- **Autenticação de usuário** para ambientes multi-tenant.
- **Testes automatizados** cobrindo os nós do grafo e as funções de leitura do banco.
