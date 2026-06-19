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

O assistente é implementado como um **agente ReAct** (*Reason + Act*) usando o **LangGraph**. Isso significa que o modelo de linguagem não segue um roteiro fixo — ele raciocina sobre a pergunta, decide quais ferramentas chamar e quantas vezes, e adapta o plano com base nas respostas que recebe.

### Como funciona o loop ReAct

```
Pergunta do usuário
        │
        ▼
┌─────────────────────────────────┐
│  LLM (Gemini 2.5 Flash)        │◄──────────────────┐
│  Decide a próxima ação         │                   │
└──────┬──────────────────────────┘                   │
       │                                              │
       ├─► Chamar get_database_schema                 │
       │        │                                     │
       │        └─► Retorna schema (tabelas/colunas)  │
       │                    │ observação              │
       │                    └─────────────────────────┘
       │
       ├─► Chamar execute_sql_query (pode chamar várias vezes)
       │        │
       │        └─► Retorna dados do SQLite em JSON
       │                    │ observação
       │                    └─────────────────────────┐
       │                                              │
       │   LLM analisa os dados e raciocina de novo ──┘
       │
       └─► Sem mais ferramentas → Resposta final em pt-BR
```

### Ferramentas disponíveis para o agente

| Ferramenta | Descrição |
|---|---|
| `get_database_schema` | Retorna a estrutura completa do banco: tabelas, colunas e tipos. O agente chama esta ferramenta primeiro, antes de construir qualquer query. |
| `execute_sql_query` | Executa uma query SELECT no SQLite (modo somente leitura) e retorna os resultados. Pode ser chamada várias vezes na mesma interação. |

### Por que ReAct e não pipeline fixo?

Em vez de um fluxo linear (`gerar SQL → executar → formatar`), o agente:
- **Explora** o banco de dados antes de assumir nomes de tabelas ou colunas.
- **Adapta** a query se ela retornar erro ou resultado inesperado.
- **Combina** múltiplas consultas quando a pergunta exige isso.
- **Decide sozinho** quando tem informação suficiente para responder.

O modelo utilizado é o **Gemini 2.5 Flash** via `langchain-google-genai`, com temperatura `0` para máxima precisão nas queries.

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

## Funcionalidades da Interface

| Funcionalidade | Descrição |
|---|---|
| **Raciocínio em tempo real** | Cada passo do agente aparece conforme acontece (schema, SQL, resultado) via `st.status()` |
| **Memória conversacional** | O histórico completo de mensagens é mantido entre perguntas — o agente entende "e desse grupo, quais...?" |
| **Perguntas sugeridas** | Após cada resposta, o agente gera 3 sugestões de perguntas de acompanhamento clicáveis |
| **Exportar CSV** | Botão para baixar os dados retornados pela última consulta |
| **Gráficos automáticos** | Barras ou linha, detectados automaticamente pelo tipo de dados |
| **Rejeição de perguntas fora do escopo** | O agente identifica perguntas não relacionadas ao banco e responde diretamente sem consultar ferramentas |

## Sugestões de Melhorias Futuras

- **Autenticação de usuário** para ambientes multi-tenant.
- **Testes automatizados** cobrindo as ferramentas e o fluxo do agente.
- **Cache de resultados** para queries repetidas dentro da mesma sessão.
