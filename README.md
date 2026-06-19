# Assistente Virtual de Dados — Desafio FRANQ

Assistente conversacional Text-to-SQL que transforma perguntas em linguagem natural em consultas SQLite, executa-as e retorna respostas claras em Português do Brasil, com interface interativa via Streamlit.

Repositório: [github.com/santossribeiroo/desafio-franq-ia](https://github.com/santossribeiroo/desafio-franq-ia.git)

---

## Sumário

1. [Pré-requisitos](#pré-requisitos)
2. [Instalação](#instalação)
3. [Configuração](#configuração)
4. [Executando o projeto](#executando-o-projeto)
5. [Arquitetura e fluxo ReAct](#arquitetura-e-fluxo-react)
6. [Funcionalidades](#funcionalidades)
7. [Exemplos de consultas](#exemplos-de-consultas)
8. [Testes automatizados](#testes-automatizados)
9. [Estrutura do projeto](#estrutura-do-projeto)

---

## Pré-requisitos

- **Python 3.11+** — [python.org/downloads](https://www.python.org/downloads/)
- **Git** — [git-scm.com](https://git-scm.com/)
- **Chave de API do Google Gemini** *(padrão recomendado)* — gratuita em [aistudio.google.com](https://aistudio.google.com)
- **Ollama** *(alternativa local, sem limite de requisições)* — [ollama.com](https://ollama.com)

---

## Instalação

```bash
# 1. Clone o repositório
git clone https://github.com/santossribeiroo/desafio-franq-ia.git
cd desafio-franq-ia

# 2. Crie e ative o ambiente virtual
python -m venv venv

# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# 3. Instale as dependências
pip install -r requirements.txt
```

---

## Configuração

Copie o arquivo de exemplo e configure o provedor de LLM:

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

### Opção A — Google Gemini (padrão, recomendado para avaliadores)

Obtenha sua chave gratuita em [aistudio.google.com](https://aistudio.google.com) → **"Get API key"**.

```env
LLM_PROVIDER=gemini
GOOGLE_API_KEY=sua-chave-aqui
GEMINI_MODEL=gemini-2.5-flash
```

### Opção B — Ollama local (sem limite de requisições)

Ideal para desenvolvimento local com hardware compatível (GPU com 8 GB+ VRAM).

```bash
# Baixe o modelo (≈9 GB, uma única vez)
ollama pull qwen2.5:14b
```

```env
LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen2.5:14b
OLLAMA_BASE_URL=http://localhost:11434
```

> O modelo `qwen2.5:14b` apresenta excelente desempenho em tarefas SQL e cabe confortavelmente em GPUs com 10 GB VRAM.

---

## Executando o projeto

```bash
# Interface Streamlit (recomendado)
streamlit run app.py
```

Acesse: [http://localhost:8501](http://localhost:8501)

```bash
# Smoke test via terminal (opcional)
python main.py
```

---

## Arquitetura e fluxo ReAct

O assistente é implementado como um **agente ReAct** (*Reason + Act*) usando **LangGraph**. O modelo raciocina sobre a pergunta, decide quais ferramentas chamar, observa os resultados e adapta o plano — sem seguir um roteiro fixo.

```
Pergunta do usuário
        │
        ▼
┌─────────────────────────────┐
│  LLM (Gemini / Ollama)      │◄──────────────────┐
│  Decide a próxima ação      │                   │
└──────┬──────────────────────┘                   │
       │                                          │
       ├─► get_database_schema                    │
       │        └─► Retorna schema completo       │
       │                 │ observação             │
       │                 └────────────────────────┘
       │
       ├─► execute_sql_query  (pode chamar N vezes)
       │        └─► Retorna dados JSON do SQLite
       │                 │ observação
       │                 └────────────────────────┐
       │                                          │
       │   LLM analisa e raciocina novamente ─────┘
       │
       └─► Resposta final em pt-BR
```

### Ferramentas disponíveis

| Ferramenta | Descrição |
|---|---|
| `get_database_schema` | Retorna schema completo (tabelas, colunas, tipos, PKs). Sempre executada primeiro. Resultado em cache por sessão. |
| `execute_sql_query` | Executa SELECT no SQLite (somente leitura). Injeta `LIMIT 200` automaticamente, valida sintaxe via `EXPLAIN QUERY PLAN`, timeout de 10s. Resultado em cache por query. |

### Por que ReAct em vez de pipeline fixo?

| Pipeline fixo | Agente ReAct |
|---|---|
| Assume nomes de tabelas | Consulta o schema antes de qualquer query |
| Para no primeiro erro | Reformula a query e tenta novamente |
| Uma query por pergunta | Combina múltiplas queries quando necessário |
| Não mantém contexto | Usa histórico para perguntas de acompanhamento |

### Provedor de LLM configurável

A fábrica `src/llm.py` lê `LLM_PROVIDER` do `.env` e retorna o modelo correto — sem alterar nenhum outro arquivo:

```
LLM_PROVIDER=gemini  →  ChatGoogleGenerativeAI (gemini-2.5-flash)
LLM_PROVIDER=ollama  →  ChatOllama (qwen2.5:14b)
```

---

## Funcionalidades

### Inteligência do agente

| Funcionalidade | Detalhes |
|---|---|
| **Loop ReAct estrito** | `langgraph.prebuilt.create_react_agent v1` — o modelo obrigatoriamente chama ferramentas antes de responder |
| **Schema-first** | `get_database_schema` é sempre a primeira ação — elimina alucinações de nomes de tabela |
| **Memória conversacional** | Histórico completo de mensagens passado ao agente — suporta perguntas de acompanhamento ("e desse grupo...?") |
| **Sumarização de contexto** | Após 50 mensagens, o histórico mais antigo é resumido automaticamente para controlar o uso de tokens |
| **Rejeição fora do escopo** | Perguntas não relacionadas ao banco são respondidas diretamente, sem chamar ferramentas |

### Interface Streamlit

| Funcionalidade | Detalhes |
|---|---|
| **Streaming em tempo real** | Cada etapa do agente (schema, SQL, resultado) aparece conforme é executada via `st.status()` |
| **Persistência de resultados** | Resultados sobrevivem a reruns da UI (troca de gráfico, zoom) via `st.session_state` |
| **Perguntas sugeridas** | 3 sugestões de acompanhamento geradas pelo LLM após cada resposta |
| **Gráficos interativos** | Barras, Linha, Pizza e Dispersão — selecionáveis sem re-executar a query |
| **Resumo estatístico** | Mín, Máx, Média, Total para colunas numéricas |
| **Detecção de outliers** | Método IQR com aviso automático |
| **Exportar CSV** | Download dos dados retornados pela última consulta |
| **Feedback** | Botões 👍/👎 persistidos em `data/feedback.json` |
| **Sessão persistente** | Histórico e feedback sobrevivem a refresh via `data/session.json` |
| **Schema na sidebar** | Árvore de tabelas/colunas com prévia de 3 linhas |
| **Exemplos clicáveis** | Welcome screen e sidebar com atalhos para consultas comuns |
| **Exportar conversa** | Download do histórico completo em Markdown |

### Segurança e robustez

| Mecanismo | Detalhes |
|---|---|
| **Somente SELECT** | Queries não-SELECT são bloqueadas antes de chegar ao SQLite |
| **Conexão read-only** | `PRAGMA query_only = ON` a cada execução |
| **Timeout de query** | 10 segundos via `sqlite3.set_progress_handler` |
| **LIMIT automático** | Injeta `LIMIT 200` se ausente — previne full-table scans |
| **Cache de schema** | Uma única leitura do DB por processo |
| **Cache de queries** | LRU com 50 entradas — evita re-execução de queries idênticas |
| **Retry automático** | Até 3 tentativas com backoff exponencial em erros 429 (Gemini) |

---

## Exemplos de consultas

| Pergunta | Tipo de análise |
|---|---|
| `Quais são os 5 clientes que mais gastaram?` | Ranking — ORDER BY + LIMIT |
| `Qual é a receita total por categoria de compra?` | Agregação — GROUP BY + SUM |
| `Qual canal de compra gerou mais receita?` | GROUP BY + SUM |
| `Quais campanhas de marketing tiveram mais interações?` | Contagem com filtro booleano |
| `Qual a porcentagem de clientes que interagiram com cada campanha?` | Cálculo proporcional com subquery |
| `Quantos chamados de suporte foram resolvidos?` | Filtro booleano |
| `Qual foi a receita total de compras por mês?` | Série temporal — GROUP BY mês |
| `E nos últimos 6 meses, qual categoria cresceu mais?` | Pergunta de acompanhamento (usa contexto) |

---

## Testes automatizados

20 testes unitários cobrindo as três camadas principais:

```bash
pytest tests/ -v
```

| Módulo | Testes |
|---|---|
| `tests/test_tools.py` | Schema caching, SQL execution, LIMIT injection, cache de queries, bloqueio não-SELECT, JOINs |
| `tests/test_llm.py` | Seleção de provider, labels, instanciação com mocks |
| `tests/test_agent.py` | Build do agente, system prompt (schema-first, português, segurança), ferramentas registradas |

---

## Estrutura do projeto

```
desafio-franq-ia/
├── app.py                  # Interface Streamlit (UI principal)
├── main.py                 # Smoke test via terminal
├── requirements.txt        # Dependências Python
├── .env.example            # Template de variáveis de ambiente
│
├── src/
│   ├── agent.py            # Agente ReAct (LangGraph)
│   ├── llm.py              # Fábrica de LLM (Gemini / Ollama)
│   ├── tools.py            # Ferramentas: get_database_schema, execute_sql_query
│   └── db_utils.py         # Conexão SQLite e utilitários de schema
│
├── tests/
│   ├── test_agent.py       # Testes do agente
│   ├── test_llm.py         # Testes do provedor de LLM
│   └── test_tools.py       # Testes das ferramentas SQL
│
├── data/
│   └── anexo_desafio_1.db  # Banco de dados SQLite (clientes, compras, suporte, campanhas)
│
└── scripts/
    └── discover_models.py  # Utilitário: lista modelos Gemini disponíveis
```

### Banco de dados

| Tabela | Colunas principais |
|---|---|
| `clientes` | id, nome, email, valor_total_gasto, data_ultima_compra, idade, cidade, estado, profissao, genero |
| `compras` | id, cliente_id, data_compra, valor, categoria, canal |
| `suporte` | id, cliente_id, data_contato, tipo_contato, resolvido, canal |
| `campanhas_marketing` | id, cliente_id, nome_campanha, data_envio, interagiu, canal |
