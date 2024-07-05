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
import folium
from streamlit_folium import st_folium
from streamlit_folium import folium_static 
from  PIL import Image
import plotly.figure_factory as ff
from datetime import datetime, timedelta
import plotly.graph_objs as go
import io
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
import re
import base64
from io import BytesIO
from reportlab.lib.units import inch
from reportlab.platypus import Image 
import plotly.io as pio
pio.kaleido.scope.default_format = "png"
from streamlit_option_menu import option_menu
from PIL import Image 
import openpyxl

# Configuração da página
st.set_page_config(page_title="CBM SF Chatbot", page_icon="🏗️", layout="wide")

# Configurações do Gemini
genai.configure(api_key="API_KEY")  
MODELO_NOME = "gemini-1.5-pro"
model = genai.GenerativeModel(MODELO_NOME)

# Carregamento dos dados (uma única vez)
@st.cache_data
def load_data():
    df_sf = pd.read_excel("SF.xlsx")
    df_sf = df_sf.fillna("Sem informação disponível.")
    return df_sf

df_sf = load_data()
NOMES_COLUNAS = df_sf.columns.tolist()

# Inicialização do estado da sessão (uma única vez)
if 'historico_mensagens' not in st.session_state:
    st.session_state.historico_mensagens = []
if 'df_resultado' not in st.session_state:
    st.session_state.df_resultado = None

def normalize_text(text):
    return unidecode(text.lower())

def identificar_saudacao(texto):
    saudacoes = ['oi', 'olá', 'ola', 'bom dia', 'boa tarde', 'boa noite', 'obrigado', 'obrigada', 'tchau', 'até logo']
    texto_lower = texto.lower()
    return any(saudacao in texto_lower for saudacao in saudacoes)

def responder_saudacao(texto):
    texto_lower = texto.lower()
    if 'bom dia' in texto_lower:
        return "Bom dia! Como posso ajudar você hoje com a planilha SF?"
    elif 'boa tarde' in texto_lower:
        return "Boa tarde! Em que posso ser útil com a planilha SF?"
    elif 'boa noite' in texto_lower:
        return "Boa noite! Estou aqui para ajudar com a planilha SF. O que você precisa?"
    elif 'obrigado' in texto_lower or 'obrigada' in texto_lower:
        return "De nada! Fico feliz em poder ajudar. Há mais alguma coisa que eu possa fazer por você com a planilha SF?"
    elif 'tchau' in texto_lower or 'até logo' in texto_lower:
        return "Até logo! Foi um prazer ajudar. Tenha um ótimo dia!"
    else:
        return "Olá! Como posso ajudar você hoje com a planilha SF?"



def buscar_dados_sf(pergunta, nome_coluna, similarity_threshold=75):
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
def analisar_oportunidades_por_valor(df, n=5, ordem='maiores'):
    """
    Analisa as oportunidades com base no Valor CBM.amount.

    Args:
    df (pd.DataFrame): DataFrame contendo os dados das oportunidades.
    n (int): Número de oportunidades a serem retornadas. Padrão é 5.
    ordem (str): 'maiores' para os maiores valores, 'menores' para os menores. Padrão é 'maiores'.

    Returns:
    pd.DataFrame: DataFrame com as n oportunidades ordenadas por Valor CBM.amount.
    str: Mensagem descritiva do resultado.
    """
    if 'Valor CBM.amount' not in df.columns:
        return pd.DataFrame(), "A coluna 'Valor CBM.amount' não foi encontrada no DataFrame."

    df_ordenado = df.sort_values(by='Valor CBM.amount', ascending=(ordem == 'menores'))
    df_resultado = df_ordenado.head(n)

    mensagem = f"As {n} {'maiores' if ordem == 'maiores' else 'menores'} oportunidades por Valor CBM:"
    
    return df_resultado, mensagem

def extrair_numero_oportunidades(pergunta):
    """
    Extrai o número de oportunidades mencionado na pergunta.
    
    Args:
    pergunta (str): A pergunta do usuário.
    
    Returns:
    int: O número de oportunidades mencionado, ou 5 se não for especificado.
    """
    match = re.search(r'\b(\d+)\s*(maiores?|menores?|primeiras?|últimas?)\b', pergunta, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return 5  # valor padrão se não for especificado



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

def identificar_coluna(pergunta, similarity_threshold=75):
    pergunta_normalizada = normalize_text(pergunta)
    pontuacoes_colunas = [fuzz.partial_ratio(pergunta_normalizada, normalize_text(coluna)) for coluna in NOMES_COLUNAS]
    
    max_pontuacao = max(pontuacoes_colunas)
    if max_pontuacao >= similarity_threshold:
        return NOMES_COLUNAS[pontuacoes_colunas.index(max_pontuacao)]
    else:
        return None

def interpretar_pergunta_matematica(pergunta):
    operacoes = {
        'soma': ['soma', 'total', 'somatório'],
        'media': ['média', 'media', 'mediana'],
        'maximo': ['máximo', 'maximo', 'maior'],
        'minimo': ['mínimo', 'minimo', 'menor']
    }
    
    pergunta_lower = pergunta.lower()
    for op, keywords in operacoes.items():
        if any(keyword in pergunta_lower for keyword in keywords):
            return op
    return None
def identificar_intencao(pergunta):
    """Identifica a intenção da pergunta, mesmo que implícita."""
    pergunta = pergunta.lower()
    intencoes = {
        "visao_geral": ["visao geral", "panorama", "resumo"],
        "valor_total": ["valor total", "soma", "quanto"],
        "distribuicao_setor": ["distribuição por setor", "dividido por setor"],
        "distribuicao_estado": ["distribuição por estado", "dividido por estado"],
        "projetos_estado": ["projetos em", "obras em", "oportunidades em"],
        # ... adicione mais intenções e suas palavras-chave
    }
    for intencao, palavras_chave in intencoes.items():
        if any(palavra in pergunta for palavra in palavras_chave):
            return intencao
    return "desconhecida"  # Se nenhuma intenção for encontrada

def gerar_analise(dados_sf, intencao, cliente=None, estado=None):
    """Gera uma análise com base na intenção, dados, cliente e estado."""
    df = pd.DataFrame(dados_sf)

    if cliente:
        df = df[df['Empresa'].str.lower() == cliente.lower()]
    if estado:
        df = df[df['Estado'].str.lower() == estado.lower()]

    if intencao == "visao_geral":
        analise = "Aqui está uma visão geral das oportunidades"
        if cliente: 
            analise += " da {}:\n\n".format(cliente)
        elif estado:
            analise += " em {}:\n\n".format(estado)
        else:
            analise += ":\n\n"

        analise += "- Valor total das oportunidades: R$ {:,.2f}\n".format(df['Valor CBM.amount'].sum())
        analise += "- Número de oportunidades: {}\n".format(len(df))
        analise += "- Setores de atuação: {}\n".format(", ".join(df['Setor'].unique()))
        # ... (adicione mais elementos para a visão geral)

    elif intencao == "valor_total":
        analise = "O valor total das oportunidades"
        if cliente:
            analise += " da {} é de R$ {:,.2f}".format(cliente, df['Valor CBM.amount'].sum())
        elif estado:
            analise += " em {} é de R$ {:,.2f}".format(estado, df['Valor CBM.amount'].sum())
        else:
            analise += " é de R$ {:,.2f}".format(df['Valor CBM.amount'].sum())

    elif intencao == "distribuicao_setor":
        analise = "Distribuição de oportunidades por setor:\n\n"
        for setor in df['Setor'].unique():
            valor_setor = df[df['Setor'] == setor]['Valor CBM.amount'].sum()
            analise += f"- {setor}: R$ {valor_setor:,.2f}\n"

    elif intencao == "distribuicao_estado":
        analise = "Distribuição de oportunidades por estado:\n\n"
        for estado in df['Estado'].unique():
            valor_estado = df[df['Estado'] == estado]['Valor CBM.amount'].sum()
            analise += f"- {estado}: R$ {valor_estado:,.2f}\n"

    elif intencao == "projetos_estado":
        if estado:
            analise = f"Projetos em {estado}:\n\n"
            for i, row in df.iterrows():
                analise += f"- {row['Nome da oportunidade']} (Valor CBM: R$ {row['Valor CBM.amount']:,.2f})\n"
        else:
            analise = "Por favor, especifique o estado para o qual deseja ver os projetos."

    else:
        analise = "Desculpe, ainda não sei como gerar essa análise. Seja mais específico."

    return analise

def create_pie_chart(df, column, title):
    if df.empty or column not in df.columns:
        st.warning(f"Dados insuficientes para criar o gráfico de pizza para {column}")
        return None
    
    fig = px.pie(df, names=column, values='Valor CBM.amount', title=title)
    return fig

import plotly.graph_objects as go
from datetime import timedelta

import plotly.graph_objects as go
from datetime import timedelta

def create_timeline_chart(df):
    """Cria um gráfico de dispersão similar a um Gantt para as oportunidades.

    Args:
        df (pd.DataFrame): O DataFrame contendo os dados das oportunidades.
    Returns:
        plotly.graph_objects._figure.Figure ou None: O gráfico de dispersão ou None se
        não for possível criar o gráfico.
    """
    if df.empty or 'Data de Assinatura' not in df.columns:
        st.warning("Dados insuficientes para criar o gráfico. "
                   "Verifique se a coluna 'Data de Assinatura' existe.")
        return None

    try:
        # Converter para datetime
        df['Data de Assinatura'] = pd.to_datetime(df['Data de Assinatura'], errors='coerce')

        # Remover linhas com datas inválidas
        df.dropna(subset=['Data de Assinatura'], inplace=True)

        if not df['Data de Assinatura'].empty:
            # Criar o gráfico
            fig = go.Figure()

            # Definir cores para cada cenário
            color_map = {'Agressivo': 'red', 'Conservador': 'blue'}

            # Gerar pontos para cada dia de cada oportunidade
            for idx, row in df.iterrows():
                data_assinatura = row['Data de Assinatura']
                data_inicio = data_assinatura - timedelta(days=240)  # Calcula a data de início

                for i in range(240):  # Itera pelos 240 dias
                    data_ponto = data_inicio + timedelta(days=i)
                    fig.add_trace(go.Scatter(
                        x=[data_ponto],
                        y=[row['Nome da oportunidade']],
                        mode='markers',
                        marker=dict(
                            size=8,
                            color=color_map.get(row['Cenário'], 'gray')
                        ),
                        name=row['Cenário'],
                        hoverinfo='text',
                        text=f"Oportunidade: {row['Nome da oportunidade']}<br>"
                             f"Data de Início: {data_inicio.strftime('%Y-%m-%d')}<br>" 
                             f"Data de Assinatura: {data_assinatura.strftime('%Y-%m-%d')}<br>"
                             f"Cenário: {row['Cenário']}"
                    ))

            # Configurar o layout
            fig.update_layout(
                title='Linha do Tempo das Oportunidades (Data de Início até Assinatura)',
                xaxis_title='Data',
                yaxis_title='Oportunidades',
                xaxis_range=[df['Data de Assinatura'].min() - timedelta(days=300),
                             df['Data de Assinatura'].max() + timedelta(days=30)],
                height=600,
                width=1000,
                showlegend=True
            )

            return fig
        else:
            st.warning("Não há datas válidas na coluna 'Data de Assinatura' "
                       "para exibir no gráfico.")
            return None
    except Exception as e:
        st.error(f"Erro ao criar o gráfico: {str(e)}")
        return None


def calcular_projetos_simultaneos(df):
    """Calcula o número de projetos simultâneos por mês.

    Args:
        df (pd.DataFrame): O DataFrame contendo 'Data de Assinatura'.
    Returns:
        pd.DataFrame: Um DataFrame com 'Mês/Ano' e 'Projetos Simultâneos'.
    """
    if df.empty or 'Data de Assinatura' not in df.columns:
        st.warning(
            "Dados insuficientes para calcular projetos simultâneos. "
            "Verifique se a coluna 'Data de Assinatura' existe."
        )
        return None

    try:
        df['Data de Assinatura'] = pd.to_datetime(
            df['Data de Assinatura'], errors='coerce'
        )
        df.dropna(subset=['Data de Assinatura'], inplace=True)

        if not df['Data de Assinatura'].empty:
            # Encontra a data mínima e máxima de assinatura
            data_min = df['Data de Assinatura'].min()
            data_max = df['Data de Assinatura'].max()

            # Cria uma lista para armazenar os dados de projetos simultâneos
            dados_projetos = []

            # Itera pelos meses entre a data mínima e máxima
            data_atual = data_min
            while data_atual <= data_max:
                mes_ano = data_atual.strftime('%Y-%m')
                projetos_no_mes = 0

                # Verifica cada oportunidade
                for _, row in df.iterrows():
                    data_inicio = row['Data de Assinatura'] - timedelta(days=240)
                    data_fim = row['Data de Assinatura']

                    # Verifica se a oportunidade está ativa no mês atual
                    if data_inicio <= data_atual < data_fim:
                        projetos_no_mes += 1

                dados_projetos.append({'Mês/Ano': mes_ano, 'Projetos Simultâneos': projetos_no_mes})
                data_atual += timedelta(days=31)  # Avança para o próximo mês

            return pd.DataFrame(dados_projetos)
        else:
            st.warning("Não há datas válidas na coluna 'Data de Assinatura'.")
            return None

    except Exception as e:
        st.error(f"Erro ao calcular projetos simultâneos: {str(e)}")
        return None



def exportar_excel(df, nome_arquivo='exportacao.xlsx'):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, sheet_name='Sheet1', index=False)
    output.seek(0)
    return output, nome_arquivo

def gerar_relatorio_pdf(resposta, df_resultado, figuras):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)

    elements = []

    # Adicionar logo
    logo_path = r"C:\Users\070283\Downloads\Logo - fundo transparente - preto.png"
    logo = Image(logo_path, width=2*inch, height=1*inch)
    elements.append(logo)
    elements.append(Spacer(1, 12))

    # Estilos
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=1))

    # Adicionar resposta do robô
    elements.append(Paragraph("Resposta do Assistente:", styles['Heading1']))
    elements.append(Paragraph(resposta, styles['Justify']))
    elements.append(Spacer(1, 12))

    # Adicionar tabela de resultados
    if df_resultado is not None and not df_resultado.empty:
        elements.append(Paragraph("Dados da Análise:", styles['Heading2']))
        data = [df_resultado.columns.tolist()] + df_resultado.values.tolist()
        table = Table(data)
        table.setStyle(TableStyle([
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
        ]))
        elements.append(table)
        elements.append(Spacer(1, 12))

    # Adicionar gráficos
    for i, figura in enumerate(figuras):
        img_buffer = BytesIO()
        # Usa write_image para salvar o gráfico Plotly como PNG
        pio.write_image(figura, img_buffer, format='png')
        img_buffer.seek(0)
        img = Image(img_buffer)
        img.drawHeight = 4*inch*img.drawHeight / img.drawWidth
        img.drawWidth = 6*inch
        elements.append(Paragraph(f"Gráfico {i+1}", styles['Heading3']))
        elements.append(img)
        elements.append(Spacer(1, 12))

    # Gerar PDF
    doc.build(elements)
    pdf = buffer.getvalue()
    buffer.close()
    return pdf


def gerar_resposta_contexto(pergunta, contexto, dados_sf=None):
    operacao_matematica = interpretar_pergunta_matematica(pergunta)

    if dados_sf and operacao_matematica:
        nome_coluna = identificar_coluna(pergunta)
        if nome_coluna:
            resultado_calculo = realizar_calculos_sf(dados_sf, nome_coluna, operacao_matematica)
            contexto += f"\n\nResultado do cálculo ({operacao_matematica}): {resultado_calculo}"
    intencao = identificar_intencao(pergunta)
    cliente = None
    estado = None

    # Tenta identificar o cliente na pergunta
    for index, row in df_sf.iterrows():
        empresa = str(row['Empresa']).lower()
        if empresa in pergunta.lower():
            cliente = row['Empresa']
            break
    
    # Tenta identificar o estado na pergunta
    for sigla in df_sf['Estado'].unique():
        if sigla.lower() in pergunta.lower():
            estado = sigla
            break

    if dados_sf:
        resposta = gerar_analise(dados_sf, intencao, cliente, estado)
    else:
        resposta = "Não encontrei informações suficientes para responder à sua pergunta. "

    prompt = f"""
    Contexto: Você é um assistente virtual especializado em responder perguntas sobre a planilha SF, que contém informações detalhadas sobre oportunidades de negócio da Construtora Barbosa Mello no setor de construção pesada no Brasil. 
    
    A planilha organiza as oportunidades em diferentes dimensões, como:
    
    * **Setor:**  Indica o setor da economia ao qual a oportunidade pertence (ex: Transporte, Mineração, Água e Saneamento).
    * **Segmento:** Detalha o segmento específico dentro do setor (ex: Rodovias, Mineração, Água e Esgoto).
    * **Fase:** Indica a fase atual da oportunidade no processo de negócio (ex: Mercado, Prospecção).
    * **Cenário:** Classifica a agressividade da busca pela oportunidade (ex: Mapeamento, Agressivo).
    * **Empreendimento:** Descreve o projeto de construção da oportunidade (ex: Rodovias do Paraná, Mina ITM2).
    * **Empresa:** Identifica a empresa cliente ou contratante da oportunidade (ex: Governo PR, Itaminas).
    * **Valor Total da Oportunidade:**  Informa o valor total do projeto de construção.
    * **% de Participação CBM:**  Indica a porcentagem de participação da Construtora Barbosa Mello no projeto.
    * **Valor CBM:** Representa o valor da participação financeira da Construtora Barbosa Mello na oportunidade.
    * **Data da Entrega Comercial:** Previsão de entrega da obra.
    * **Status do projeto:** Situação atual da oportunidade (ex: Prospecção, N/A).
    * **Estado:**  Sigla da unidade federativa onde se localiza a oportunidade.

    Informações relevantes:
    {contexto}


    Pergunta do usuário: {pergunta}

    Observações: 
    - Se a resposta gerada já for satisfatória, apenas refine a linguagem para torná-la mais natural e profissional.
    - Se necessário, complemente a resposta com informações de mercado. 
    """
    if dados_sf:
        prompt += f"\n\nDados do SF utilizados:\n{dados_sf}"

    if dados_sf:
        prompt += f"\n\nDados do SF utilizados:\n{dados_sf}"

    prompt += """
    Por favor, siga estas diretrizes para formular sua resposta:
    
    1. **Compreensão Profunda:** Analise a pergunta do usuário e utilize todos os dados relevantes da planilha SF para formular uma resposta completa e informativa.
    2. **Análise Detalhada:** Vá além de simplesmente apresentar os dados. Interprete as informações, identifique relações entre os diferentes campos da planilha e extraia insights relevantes.
    3. **Contexto da CBM:** Utilize seu conhecimento sobre a Construtora Barbosa Mello e o mercado de construção pesada no Brasil para contextualizar as informações e gerar análises aprofundadas.
    4. **Comparativo com o Mercado:**  Sempre que possível, compare os dados da planilha com o cenário geral do mercado de construção no Brasil, destacando similaridades, diferenças e tendências relevantes.
    5. **Linguagem Clara e Objetiva:** Apresente a resposta de forma clara, concisa e profissional, utilizando linguagem adequada ao contexto da análise de dados de negócios.
    
    Lembre-se: 
    *  Você não deve inventar informações. Baseie suas respostas apenas nos dados fornecidos e em seu conhecimento prévio. 
    *  Se a pergunta do usuário for ambígua ou precisar de mais detalhes, solicite gentilmente mais informações."""

    resposta = model.generate_content(prompt)
    return resposta.text
    resposta = model.generate_content(prompt)
    return resposta.text
# Declara historico_mensagens no escopo global, inicializando como uma lista vazia
historico_mensagens = []  

def processar_pergunta(pergunta):
    if identificar_saudacao(pergunta):
        return responder_saudacao(pergunta), None, None, None, None, None

    nome_coluna = identificar_coluna(pergunta)
    dados_sf = None
    contexto = ""
    df_resultado = None
    excel_file = None
    excel_filename = None
    pdf_file = None
    pdf_filename = None

    if nome_coluna:
        dados_sf = buscar_dados_sf(pergunta, nome_coluna)
        if dados_sf:
            df_resultado = pd.DataFrame(dados_sf)
            
            # Análise de maiores/menores oportunidades
            n = extrair_numero_oportunidades(pergunta)
            if "maiores oportunidades" in pergunta.lower() or "top oportunidades" in pergunta.lower():
                df_analise, mensagem = analisar_oportunidades_por_valor(df_resultado, n, 'maiores')
                contexto += f"\n\n{mensagem}\n{df_analise.to_string()}"
            elif "menores oportunidades" in pergunta.lower():
                df_analise, mensagem = analisar_oportunidades_por_valor(df_resultado, n, 'menores')
                contexto += f"\n\n{mensagem}\n{df_analise.to_string()}"
            
            # Outras análises existentes...
            operacao_matematica = interpretar_pergunta_matematica(pergunta)
            if operacao_matematica:
                resultado_calculo = realizar_calculos_sf(dados_sf, nome_coluna, operacao_matematica)
                contexto += f"\nResultado do cálculo ({operacao_matematica}): {resultado_calculo}"
        else:
            contexto = f"Não foram encontrados dados relevantes na coluna '{nome_coluna}'."
    else:
        contexto = "Não foi possível identificar uma coluna relevante na planilha SF para sua pergunta."

    resposta = gerar_resposta_contexto(pergunta, contexto, dados_sf)

    return resposta, df_resultado, excel_file, excel_filename, pdf_file, pdf_filename

    # Adiciona a pergunta e a resposta ao histórico
    historico_mensagens = [
        {"role": "user", "content": pergunta},
        {"role": "assistant", "content": resposta}
    ]

    return historico_mensagens, df_resultado, excel_file, excel_filename, pdf_file, pdf_filename



# Configuração do Streamlit

# Inicialização do estado da sessão
if 'historico_mensagens' not in st.session_state:
    st.session_state.historico_mensagens = []
if 'df_resultado' not in st.session_state:
    st.session_state.df_resultado = None
if 'figura' not in st.session_state:
    st.session_state.figura = None
if 'mapa' not in st.session_state:
    st.session_state.mapa = None
if 'excel_file' not in st.session_state:
    st.session_state.excel_file = None
if 'excel_filename' not in st.session_state:
    st.session_state.excel_filename = None
if 'pdf_file' not in st.session_state:
    st.session_state.pdf_file = None
if 'pdf_filename' not in st.session_state:
    st.session_state.pdf_filename = None

# Carregamento dos dados
df_sf = load_data()



# Inicializar o histórico de mensagens na sessão
if "historico_mensagens" not in st.session_state:
    st.session_state.historico_mensagens = []

# Inicializar DataFrame de resultados na sessão
if "df_resultado" not in st.session_state:
    st.session_state.df_resultado = None

# Carregamento do logo da CBM
logo_path ="Logo - fundo transparente - preto.png"
logo = Image.open(logo_path)

# Cores da paleta CBM
cor_principal = "#294E88"
cor_secundaria = "#547EA7"

# CSS Personalizado para o Chat
st.markdown(
    """
    <style>
        .main {
            background-color: #f0f2f6;
        }
        .stButton>button {
            background-color: #0e1117;
            color: white;
            border-radius: 5px;
        }
        .stTextInput>div>div>input {
            background-color: #ffffff;
        }
        /* Estilos para o Chat */
        .user-message {
            background-color: white;
            color: black;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 5px;
            float: left;
            clear: both;
            max-width: 70%;
        }
        .assistant-message {
            background-color: """ + cor_principal + """; 
            color: white;
            padding: 10px;
            border-radius: 10px;
            margin-bottom: 5px;
            float: right;
            clear: both;
            max-width: 70%;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

def main():
   # Barra Lateral com o logo da CBM
    with st.sidebar:
        st.image(logo, width=150)  # Adiciona o logo na barra lateral
        # ... (resto do código da barra lateral)

        st.title("CBM SF Chatbot")
        selected = option_menu(
            menu_title="Menu",
            options=["Chat", "Análise", "Sobre"],
            icons=["chat-dots", "graph-up", "info-circle"],
            menu_icon="cast",
            default_index=0,
        )

    # Conteúdo principal
    if selected == "Chat":
        chat_interface()
    elif selected == "Análise":
        analysis_interface()
    elif selected == "Sobre":
        about_interface()

def chat_interface():
    st.header("💬 Barbosa Force")

    # Histórico de mensagens
    for message in st.session_state.historico_mensagens:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message">{message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="assistant-message">{message["content"]}</div>', unsafe_allow_html=True)

    # Entrada do usuário
    user_input = st.chat_input("Digite sua pergunta sobre a planilha SF:")
    if user_input:
        process_user_input(user_input)

def process_user_input(user_input):
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state.historico_mensagens.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"), st.spinner("Processando..."):
        resposta, df_resultado, _, _, _, _ = processar_pergunta(user_input) 
        st.markdown(resposta)
        st.session_state.df_resultado = df_resultado

        if df_resultado is not None:
            st.subheader("Resultados da Análise")
            st.dataframe(df_resultado)
            display_charts(df_resultado)

    st.session_state.historico_mensagens.append({"role": "assistant", "content": resposta})

# ... (Funções display_charts, create_pie_chart, create_timeline_chart, calcular_projetos_simultaneos, 
# gerar_relatorio_pdf, processar_pergunta - Você precisa implementá-las de acordo com suas necessidades)

def analysis_interface():
    st.header("📊 Análise de Dados")
    
    if st.session_state.df_resultado is not None:
        st.subheader("Dados Atuais")
        st.dataframe(st.session_state.df_resultado)
        display_charts(st.session_state.df_resultado)
    else:
        st.info("Nenhum dado disponível para análise. Use o chat para gerar dados.")
# Carregamento do logo da CBM
logo_path = r"C:\Users\070283\Downloads\Logo - fundo transparente - preto.png"
logo = Image.open(logo_path)

# Cores da paleta CBM
cor_principal = "#294E88"
cor_secundaria = "#547EA7"

def display_charts(df):
    """Exibe gráficos com base no DataFrame df, usando a paleta de cores da CBM."""
    if df is not None:
        st.subheader("Visualizações")

        # Gráfico de pizza para 'Setor'
        fig_setor = px.pie(
            df, 
            names='Setor', 
            values='Valor CBM.amount', 
            title='Distribuição por Setor',
            color_discrete_sequence=[cor_principal, cor_secundaria],  # Aplica cores da paleta
        )
        fig_setor.update_layout(
            title_x=0.5,  # Centraliza o título do gráfico
            font=dict(family="Arial", size=12),  # Define fonte padrão
            legend=dict(
                orientation="h",  # Orientação da legenda: horizontal
                yanchor="bottom",  # Posição vertical: inferior
                y=1.02,  # Posição vertical precisa
                xanchor="right",  # Posição horizontal: direita
                x=1  # Posição horizontal precisa
            )
        )
        st.plotly_chart(fig_setor, use_container_width=True)

        # Gráfico de pizza para 'Estado' (mesma lógica do gráfico de setor)
        fig_estado = px.pie(
            df,
            names='Estado',
            values='Valor CBM.amount',
            title='Distribuição por Estado',
            color_discrete_sequence=[cor_principal, cor_secundaria]
        )
        fig_estado.update_layout(
            title_x=0.5,
            font=dict(family="Arial", size=12),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )

        # Gráfico de linha do tempo (Gantt)
        timeline_chart = create_timeline_chart(df)
        if timeline_chart:
            st.plotly_chart(timeline_chart, use_container_width=True)

    # Projetos simultâneos (DataFrame)
    # Projetos simultâneos (DataFrame)
    df_projetos_simultaneos = calcular_projetos_simultaneos(df)
    if df_projetos_simultaneos is not None:
        st.subheader('Número de Projetos Simultâneos por Mês')
        st.dataframe(df_projetos_simultaneos.transpose()) # Transpõe o DataFrame

def about_interface():
    st.header("ℹ️ Sobre o CBM SF Chatbot")
    st.write("""
    O CBM SF Chatbot é uma ferramenta de inteligência artificial desenvolvida para auxiliar 
    na análise e interpretação dos dados da planilha SF da Construtora Barbosa Mello.

    ### Recursos:
    - Análise de dados em tempo real
    - Geração de gráficos e visualizações
    - Respostas personalizadas às suas perguntas
    - Exportação de relatórios em PDF

    ### Como usar:
    1. Acesse a aba "Chat"
    2. Digite sua pergunta sobre a planilha SF
    3. Explore os resultados e visualizações gerados
    4. Use a aba "Análise" para uma visão mais detalhada dos dados

    Para mais informações, entre em contato com o departamento de TI.
    """)
if __name__ == "__main__":
    main()
