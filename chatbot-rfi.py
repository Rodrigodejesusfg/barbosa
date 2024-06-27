import streamlit as st
import pandas as pd
import google.generativeai as genai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer, StandardScaler
from sklearn.cluster import KMeans
import numpy as np
from fuzzywuzzy import fuzz
from unidecode import unidecode
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
import io
import xlsxwriter
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph

# Configurações do Gemini
genai.configure(api_key="AIzaSyCtj2xGASpn_FrNYW9D-Nbt_F8-CXpFypQ")  
MODELO_NOME = "gemini-1.5-pro"
model = genai.GenerativeModel(MODELO_NOME)

def init_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""
    if 'sugestoes' not in st.session_state:
        st.session_state.sugestoes = []

init_session_state()

@st.cache_data
def load_data():
    df_rfi = pd.read_excel(r"C:\Users\070283\OneDrive - Construtora Barbosa Mello SA\python\projeto barbosa ai\DADOS\rfi chatbot.xlsx")
    df_rfi = df_rfi.dropna(subset=['Pergunta'])
    df_rfi['Pergunta'] = df_rfi['Pergunta'].astype(str)
    df_rfi['Resposta'] = df_rfi['Resposta'].fillna("Sem resposta disponível.").astype(str)

    df_sf = pd.read_excel(r"C:\Users\070283\OneDrive - Construtora Barbosa Mello SA\python\projeto barbosa ai\DADOS\SF.xlsx")
    df_sf = df_sf.fillna("Sem informação disponível.")

    return df_rfi, df_sf

df_rfi, df_sf = load_data()

NOMES_COLUNAS = [
    "ID Interno da Oportunidade", "Setor", "Segmento", "Fase", 
    "Tipo de registro de oportunidade", "Cenário", "Empreendimento", 
    "Empresa", "Nome da oportunidade", "Valor Total da Oportunidade.amount", 
    "% de Participação CBM", "Valor CBM.amount", "Data da Entrega Comercial", 
    "Data de Assinatura", "Data Inicio Obra", "Prazo de Execução (dias)", 
    "Data de criação", "Projeto foco?", "Considerar projeto para faturamento", 
    "Engenheiro Orçamentista", "Gerente Comercial", "Status do projeto", "Estado"
]

# Vetorização (TF-IDF)
stop_words = ["a", "e", "o", "os", "as", "de", "do", "da", "em", "para", "com", "é", "foi", "por", "que"]  # Personalize a lista
vectorizer = TfidfVectorizer(stop_words=stop_words, min_df=2, max_df=0.8) 
tfidf_matrix = vectorizer.fit_transform(df_rfi["Pergunta"])

# Redução de Dimensionalidade (LSA)
n_components = 50  # Ajuste este valor
lsa = TruncatedSVD(n_components=n_components)
lsa_matrix = lsa.fit_transform(tfidf_matrix)

normalizer = Normalizer(copy=False)
lsa_matrix_norm = normalizer.fit_transform(lsa_matrix)

n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters)
df_rfi['Topico'] = kmeans.fit_predict(lsa_matrix_norm)

def normalize_text(text):
    return unidecode(text.lower())

def identificar_saudacao(texto):
    saudacoes = ['oi', 'olá', 'ola', 'bom dia', 'boa tarde', 'boa noite', 'obrigado', 'obrigada', 'tchau', 'até logo']
    texto_lower = texto.lower()
    for saudacao in saudacoes:
        if saudacao in texto_lower:
            return True
    return False

def responder_saudacao(texto):
    texto_lower = texto.lower()
    if 'bom dia' in texto_lower:
        return "Bom dia! Como posso ajudar você hoje?"
    elif 'boa tarde' in texto_lower:
        return "Boa tarde! Em que posso ser útil?"
    elif 'boa noite' in texto_lower:
        return "Boa noite! Estou aqui para ajudar. O que você precisa?"
    elif 'obrigado' in texto_lower or 'obrigada' in texto_lower:
        return "De nada! Fico feliz em poder ajudar. Há mais alguma coisa que eu possa fazer por você?"
    elif 'tchau' in texto_lower or 'até logo' in texto_lower:
        return "Até logo! Foi um prazer ajudar. Tenha um ótimo dia!"
    else:
        return "Olá! Como posso ajudar você hoje em relação à Construtora Barbosa Mello SA?"

def encontrar_perguntas_similares(pergunta_usuario, limite_similaridade=0.3, num_sugestoes=3):
    vetor_pergunta = vectorizer.transform([pergunta_usuario])
    lsa_vetor = lsa.transform(vetor_pergunta)
    lsa_vetor_norm = normalizer.transform(lsa_vetor)

    similaridades = cosine_similarity(lsa_vetor_norm, lsa_matrix_norm).flatten()
    indices_similares = similaridades.argsort()[::-1]

    perguntas_similares = []
    for idx in indices_similares:
        if similaridades[idx] > limite_similaridade:
            perguntas_similares.append((df_rfi.iloc[idx]['Pergunta'], df_rfi.iloc[idx]['Resposta'], similaridades[idx]))
        if len(perguntas_similares) >= num_sugestoes:
            break

    return perguntas_similares
def encontrar_resposta_existente(pergunta_usuario, limite_similaridade=0.6):
    """
    Procura por uma resposta existente no df_rfi com base na similaridade da pergunta.
    
    Args:
        pergunta_usuario (str): A pergunta do usuário.
        limite_similaridade (float): O limite de similaridade para considerar uma resposta como existente.
    
    Returns:
        str or None: A resposta existente se encontrada, caso contrário, None.
    """
    vetor_pergunta = vectorizer.transform([pergunta_usuario])
    lsa_vetor = lsa.transform(vetor_pergunta)
    lsa_vetor_norm = normalizer.transform(lsa_vetor)

    similaridades = cosine_similarity(lsa_vetor_norm, lsa_matrix_norm).flatten()
    indice_mais_similar = similaridades.argmax()

    if similaridades[indice_mais_similar] > limite_similaridade:
        return df_rfi.iloc[indice_mais_similar]['Resposta']
    else:
        return None
def buscar_dados_sf(pergunta, nome_coluna, similarity_threshold=70):
    resultados = []
    pergunta_normalizada = normalize_text(pergunta)
    
    for index, row in df_sf.iterrows():
        valor_coluna = str(row[nome_coluna])
        valor_coluna_normalizado = normalize_text(valor_coluna)
        
        similarity = fuzz.partial_ratio(pergunta_normalizada, valor_coluna_normalizado)
        
        if similarity >= similarity_threshold:
            resultado = row.to_dict()
            resultado['similarity'] = similarity
            resultados.append(resultado)
    
    return sorted(resultados, key=lambda x: x['similarity'], reverse=True)

def realizar_calculos_sf(dados, nome_coluna, operacao):
    if not dados:
        return "Não foram encontrados dados relevantes para realizar o cálculo."

    df_calculo = pd.DataFrame(dados)

    try:
        df_calculo[nome_coluna] = pd.to_numeric(df_calculo[nome_coluna])
    except ValueError:
        return f"A coluna '{nome_coluna}' não contém dados numéricos válidos."

    if operacao == 'soma':
        resultado = df_calculo[nome_coluna].sum()
    elif operacao == 'media':
        resultado = df_calculo[nome_coluna].mean()
    elif operacao == 'maximo':
        resultado = df_calculo[nome_coluna].max()
    elif operacao == 'minimo':
        resultado = df_calculo[nome_coluna].min()
    else:
        return "Operação não reconhecida."

    return resultado

def gerar_resposta_contexto(pergunta, contexto, perguntas_similares, topico, dados_sf=None):
    operacao_matematica = interpretar_pergunta_matematica(pergunta)

    if dados_sf and operacao_matematica:
        nome_coluna = identificar_coluna(pergunta)  # Certifique-se de que esta função está definida
        if nome_coluna:
            resultado_calculo = realizar_calculos_sf(dados_sf, nome_coluna, operacao_matematica)
            contexto += f"\n\nResultado do cálculo ({operacao_matematica}): {resultado_calculo}"

def identificar_coluna(pergunta, similarity_threshold=70):
    pergunta_normalizada = normalize_text(pergunta)
    pontuacoes_colunas = [fuzz.partial_ratio(pergunta_normalizada, normalize_text(coluna)) for coluna in NOMES_COLUNAS]
    
    max_pontuacao = max(pontuacoes_colunas)
    if max_pontuacao >= similarity_threshold:
        return NOMES_COLUNAS[pontuacoes_colunas.index(max_pontuacao)]
    else:
        return None

def interpretar_pergunta_matematica(pergunta):
    operacoes = {
        'soma': ['soma', 'total', 'somar'],
        'media': ['média', 'media', 'promedio'],
        'maximo': ['máximo', 'maximo', 'maior'],
        'minimo': ['mínimo', 'minimo', 'menor']
    }

    for op, palavras_chave in operacoes.items():
        if any(palavra in pergunta.lower() for palavra in palavras_chave):
            return op

    return None

def analisar_tendencias_temporais(df, coluna_data, coluna_valor, periodo='mensal'):
    df[coluna_data] = pd.to_datetime(df[coluna_data])
    df = df.sort_values(coluna_data)
    
    if periodo == 'diario':
        df_grouped = df.groupby(df[coluna_data].dt.date)
    elif periodo == 'semanal':
        df_grouped = df.groupby(df[coluna_data].dt.to_period('W'))
    elif periodo == 'mensal':
        df_grouped = df.groupby(df[coluna_data].dt.to_period('M'))
    elif periodo == 'anual':
        df_grouped = df.groupby(df[coluna_data].dt.year)
    else:
        raise ValueError("Período inválido. Use 'diario', 'semanal', 'mensal' ou 'anual'.")
    
    tendencia = df_grouped[coluna_valor].agg(['sum', 'mean', 'count']).reset_index()
    tendencia.columns = ['Periodo', 'Soma', 'Media', 'Contagem']
    
    return tendencia

def segmentar_clientes(df):
    # Selecionar colunas relevantes para segmentação
    colunas_segmentacao = ['Valor Total da Oportunidade.amount', 'Prazo de Execução (dias)']
    
    # Preparar os dados para clustering
    X = df[colunas_segmentacao].fillna(0)
    
    # Normalizar os dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Aplicar K-means
    kmeans = KMeans(n_clusters=3, random_state=42)
    df['Segmento'] = kmeans.fit_predict(X_scaled)
    
    return df

def analisar_empresa(df, empresa):
    df_empresa = df[df['Empresa'] == empresa]
    
    if df_empresa.empty:
        return "Empresa não encontrada na base de dados."
    
    analise = {
        'total_oportunidades': len(df_empresa),
        'valor_total': df_empresa['Valor Total da Oportunidade.amount'].sum(),
        'valor_medio': df_empresa['Valor Total da Oportunidade.amount'].mean(),
        'maior_oportunidade': df_empresa['Valor Total da Oportunidade.amount'].max(),
        'segmentos': df_empresa['Segmento'].value_counts().to_dict()
    }
    
    return analise

def criar_grafico(df, x, y, tipo='bar'):
    if tipo == 'bar':
        fig = px.bar(df, x=x, y=y, title=f'{y} por {x}')
    elif tipo == 'line':
        fig = px.line(df, x=x, y=y, title=f'{y} ao longo do {x}')
    elif tipo == 'scatter':
        fig = px.scatter(df, x=x, y=y, title=f'Relação entre {x} e {y}')
    else:
        raise ValueError("Tipo de gráfico não suportado.")
    
    fig.update_layout(height=300, width=400)
    return fig

def exportar_excel(df, nome_arquivo='exportacao.xlsx'):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Sheet1', index=False)
    output.seek(0)
    return output, nome_arquivo

# Função para exportar o histórico para PDF
def exportar_pdf(ultima_resposta, df=None, nome_arquivo='exportacao.pdf'):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    styles = getSampleStyleSheet()

    # Adicionar a última resposta do chatbot ao PDF
    texto_resposta = f"ASSISTANT: {ultima_resposta}\n\n"
    elements.append(Paragraph(texto_resposta, styles['Normal']))

    # Adicionar dataframe ao PDF, se existir
    if df is not None:
        data = [df.columns.tolist()] + df.values.tolist()
        tabela = Table(data)
        estilo = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 12),
            ('TOPPADDING', (0, 1), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ])
        tabela.setStyle(estilo)
        elements.append(tabela)

    doc.build(elements)
    buffer.seek(0)
    return buffer, nome_arquivo

def gerar_resposta_contexto(pergunta, contexto, perguntas_similares, topico, dados_sf=None):
    operacao_matematica = interpretar_pergunta_matematica(pergunta)

    if dados_sf and operacao_matematica:
        resultado_calculo = realizar_calculos_sf(dados_sf, operacao_matematica)
        contexto += f"\n\nResultado do cálculo ({operacao_matematica}): {resultado_calculo}"

    perguntas_similares_texto = "\n".join([f"- {p[0]}" for p in perguntas_similares])

    prompt = f"""
    Contexto: Você é um assistente virtual especializado em responder perguntas sobre a empresa Construtora Barbosa Mello SA.
    Use as informações fornecidas abaixo para responder à pergunta do usuário. Se a informação não for suficiente,
    use seu conhecimento geral para dar uma resposta apropriada, mas mantenha o foco na empresa.

    Informações relevantes:
    {contexto}

    Pergunta do usuário: {pergunta}
    Tópico classificado: {topico}

    Perguntas similares encontradas na base de conhecimento:
    {perguntas_similares_texto}
    """

    if dados_sf:
        prompt += f"\n\nDados do SF utilizados:\n{dados_sf}"

    prompt += """
    Por favor, forneça uma resposta detalhada e profissional em formato de planilha, considerando as perguntas similares encontradas e o tópico classificado.
    Se a pergunta for ambígua ou necessitar de mais informações, indique isso na resposta e sugira que o usuário forneça mais detalhes.
    Ao final da resposta, sugira ao usuário que ele pode obter mais informações consultando as perguntas similares listadas acima.

    Se foram utilizados dados do SF, mencione especificamente quais dados foram usados para chegar à conclusão.
    Não invente informações. Se não houver dados suficientes, indique claramente que a informação não está disponível.
    """

    resposta = model.generate_content(prompt)
    return resposta.text

#  Função principal para processar a pergunta do usuário
def processar_pergunta(pergunta):
    # 0. Verificar se a pergunta é uma saudação
    if identificar_saudacao(pergunta):
        return responder_saudacao(pergunta), [], None, None, None, None, None

    # 1. Buscar resposta existente
    resposta_existente = encontrar_resposta_existente(pergunta)
    if resposta_existente:
        return resposta_existente, [], None, None, None, None, None # Retorna a resposta existente


    # 2. Identificar a coluna relevante no df_sf
    nome_coluna = identificar_coluna(pergunta)
    dados_sf = None
    contexto = ""
    df_resultado = None


    # 3. Realizar análises específicas com base na pergunta
    if "tendência" in pergunta.lower() or "tendencia" in pergunta.lower():
        # Indentação correta - 4 espaços
        coluna_data = "Data de criação"  
        coluna_valor = "Valor Total da Oportunidade.amount" 
        tendencia = analisar_tendencias_temporais(df_sf, coluna_data, coluna_valor)
        contexto = f"Análise de tendência para {coluna_valor}:\n{tendencia.to_string()}"
        df_resultado = tendencia
    elif "segmentar clientes" in pergunta.lower():
        # Indentação correta - 4 espaços
        df_segmentado = segmentar_clientes(df_sf)
        contexto = f"Segmentação de clientes realizada. Resumo:\n{df_segmentado['Segmento'].value_counts()}"
        df_resultado = df_segmentado
    elif "análise da empresa" in pergunta.lower() or "analise da empresa" in pergunta.lower():
        # Indentação correta - 4 espaços
        empresa = pergunta.split("empresa")[-1].strip()
        analise = analisar_empresa(df_sf, empresa)
        contexto = f"Análise da empresa {empresa}:\n{analise}"
        df_resultado = pd.DataFrame([analise])
    elif nome_coluna: 
        # Indentação correta - 4 espaços
        dados_sf = buscar_dados_sf(pergunta, nome_coluna)
        if dados_sf:
            operacao_matematica = interpretar_pergunta_matematica(pergunta)
            if operacao_matematica:
                resultado_calculo = realizar_calculos_sf(dados_sf, nome_coluna, operacao_matematica)
                contexto = f"Dados do SF encontrados na coluna '{nome_coluna}':\n{dados_sf}\n\nResultado do cálculo ({operacao_matematica}): {resultado_calculo}"
            else:
                contexto = f"Dados do SF encontrados na coluna '{nome_coluna}':\n{dados_sf}"
            df_resultado = pd.DataFrame(dados_sf)
    else:
        # Indentação correta - 4 espaços
        contexto = "Não foi possível identificar uma coluna relevante no sistema SF para sua pergunta."


    # 4. Encontrar perguntas similares na base de conhecimento
    perguntas_similares = encontrar_perguntas_similares(pergunta)
    topico = None  # Você pode implementar a identificação de tópico se necessário

    # 5. Gerar resposta usando o modelo de linguagem (Gemini)
    resposta = gerar_resposta_contexto(pergunta, contexto, perguntas_similares, topico, dados_sf)

    # 6. Gerar arquivos para download (se aplicável)
    if df_resultado is not None:
        excel_file, excel_filename = exportar_excel(df_resultado)
        pdf_file, pdf_filename = exportar_pdf(df_resultado)
        return resposta, perguntas_similares, df_resultado, excel_file, excel_filename, pdf_file, pdf_filename
    else:
        return resposta, perguntas_similares,  df_resultado, None, None, None, None

    # Criar downloads se df_resultado existir
    if df_resultado is not None:
        excel_file, excel_filename = exportar_excel(df_resultado)
        pdf_file, pdf_filename = exportar_pdf(df_resultado)

        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                label="Baixar Excel",
                data=excel_file,
                file_name=excel_filename,
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        with col2:
            st.download_button(
                label="Baixar PDF",
                data=pdf_file,
                file_name=pdf_filename,
                mime="application/pdf"
            )

    return resposta, sugestoes, df_resultado
    sugestoes = [p[0] for p in perguntas_similares]
    return resposta, sugestoes, df_resultado
    if nome_coluna:
        dados_sf = buscar_dados_sf(pergunta, nome_coluna)
        
        if dados_sf:
            operacao_matematica = interpretar_pergunta_matematica(pergunta)
            if operacao_matematica:
                resultado_calculo = realizar_calculos_sf(dados_sf, nome_coluna, operacao_matematica)
                contexto = f"Dados do SF encontrados na coluna '{nome_coluna}':\n{dados_sf}\n\nResultado do cálculo ({operacao_matematica}): {resultado_calculo}"
            else:
                contexto = f"Dados do SF encontrados na coluna '{nome_coluna}':\n{dados_sf}"
            df_resultado = pd.DataFrame(dados_sf)

# --- Interface do Streamlit ---
st.title("Chatbot Inteligente da Construtora Barbosa Mello SA")

# Inicializar o estado da sessão
init_session_state()

# Exibir histórico de mensagens
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Obter input do usuário
user_input = st.chat_input("Digite sua pergunta sobre a empresa:", key="chat_input")

# Exibir sugestões de perguntas (se houver)
if st.session_state.sugestoes:
    st.write("Clique em uma sugestão para fazer a pergunta:")
    cols = st.columns(len(st.session_state.sugestoes))
    for idx, sugestao in enumerate(st.session_state.sugestoes):
        if cols[idx].button(sugestao[0], key=f"sugestao{idx}"):
            user_input = sugestao[0]

# Processar a pergunta do usuário
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Processando sua pergunta..."):
            resposta, sugestoes, df_resultado, excel_file, excel_filename, pdf_file, pdf_filename = processar_pergunta(user_input)
        
        st.markdown(resposta)

        # Exibir dataframe, gráfico e botões de download (se aplicável)
        if df_resultado is not None:
            st.dataframe(df_resultado)
            if len(df_resultado.columns) >= 2:
                x_col = df_resultado.columns[0]
                y_col = df_resultado.columns[1]
                fig = criar_grafico(df_resultado, x_col, y_col)
                st.plotly_chart(fig)

            if excel_file:
                col1, col2 = st.columns(2)
                with col1:
                    st.download_button(
                        label="Baixar Excel",
                        data=excel_file,
                        file_name=excel_filename,
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                with col2:
                    st.download_button(
                        label="Baixar PDF",
                        data=pdf_file,
                        file_name=pdf_filename,
                        mime="application/pdf"
                    )

        st.session_state.sugestoes = sugestoes
    st.session_state.messages.append({"role": "assistant", "content": resposta})