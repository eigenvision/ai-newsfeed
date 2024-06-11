
#### IMPORTS ######################################################################################
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import plotly.graph_objs as go
import openai

from llama_index.core import Settings
from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.core import SimpleDirectoryReader
from llama_index.core import StorageContext, load_index_from_storage

# OTHER IMPORTS
import os
import time



#### PAGE CONFIG ######################################################################################
st.set_page_config(layout="wide")



#### IMPORT DATA ######################################################################################
# Specify the path to the Excel file
# excel_file_path = r"C:\Users\Jeff\Dropbox\Jeff DropBox\Cloud Assurance\LLM Projects\Newsfeed projects\Newsfeed streamlit apps\ai-newsfeed\AI_media_Jan_June_24_streamlit.xlsx"
excel_file_path = './AI_media_Jan_June_24_streamlit.xlsx'

# Read the Excel file into a pandas DataFrame
df = pd.read_excel(excel_file_path)

#### SYSTEM PROMPTS ##################################################

depth_prompt = """
        You are a very intelligent, very thoughtful, and helpful expert with deep technical and policy 
        knowledge AI policy and regulatory issues around the world.
        Your job is to provide maximally complete, detailed, logical, clear, and accurate answers to questions 
        about:
        1. The major trends in AI policy and regulatory developments described in the information you have
        access to.
        2. Specific facts and events regarding the relation of AI to topics such as the EU AI Act, antitrust concerns 
        raised by regulators about companies involved in AI, copyright and intellectual property,
        deepfakes and disinformation and synthetic media, Microsoft, and environmental sustainability.
        You always pay close attention to the exact phrasing of the user's question and you always 
        deliver an answer that matches every specific detail of the user's expressed intention.
        You always give the fullest and most complete answers possible in substantial paragraphs 
        full of relevant details, concrete examples, and logical analysis.
        You never make up facts. You are not lazy and you do not skimp on important context.
        You always write in very polished and clear business prose, such as might be published
        in a leading business periodical like Harvard Business Review.
        The users of this chatbot are making critical business decisions based on 
        your answers, and they need the highest quality answers possible that comply to the letter
        with their precise instructions.
        Finally, we have observed in your recent answers a distinct tendency to be lazy by providing
        answers that are short, lacking in extensive detail, and lacking in substantive analysis. It is 
        imperative that in this answer you overcome your instinct toward laziness and that you provide
        the richest, most detailed, most extensive answer that is possible to assemble from your
        sources, while still scrupulous respecting user instructions regarding the length of your response.
        YOU NEVER GIVE SHORT ANSWERS UNLESS SPECIFICALLY INSTRUCTED TO DO SO. In most cases a one
        paragraph answer is TOO SHORT.
        """
    
quick_prompt = """
        You are a quick-witted and helpful expert with technical and policy 
        knowledge AI policy and regulatory issues around the world.
        Your job is to provide quck and concise answers to questions about AI policy and regulatory 
        issues around the world.
        You pay close attention to the phrasing of the user's question.
        You never make up facts.
        You always provide answers that are quick and too the point, without unecessary
        explanations or words.
        You always write in polished and clear business prose, such as might be published
        in a leading business periodical like Harvard Business Review.
        DO NOT explicitly mention the conversation you are engaged in. Just aanswer the user's question.
        """

#### FORMAT DATA ######################################################################################
# Convert the date strings to datetime objects for better formatting
df['Date'] = pd.to_datetime(df['Date'])

# Fill NaN values in the 'Score' column with a default value, e.g., 0
df['Score'] = df['Score'].fillna(0).astype(int)


#### SET UP SIDEBAR MENUS & RADIO BUTTONS ###############################################################

# 1. Filter by Publication
publications = df['Publication'].unique()
selected_publication = st.sidebar.selectbox("Select Publications", ["All"] + list(publications))

if selected_publication != "All":
    df = df[df['Publication'] == selected_publication]

# 2. Filter by Topic
topic_labels = ['All', 'AI Act', 'Antitrust', 'Copyright', 'Deepfakes-Disinformation', 'Microsoft', 'Sustainability']
selected_topic = st.sidebar.selectbox("Select Topic", topic_labels)

if selected_topic != "All":
    df = df[df[selected_topic] == True]

# 3. Chatbot Answer Style
style_settings = {
    "Short answers": {
        "s_prompt": quick_prompt,
        "model": "gpt-4-turbo"
    },
    "In-depth answers": {
        "s_prompt": depth_prompt,
        "model": "gpt-4-turbo"
    }
}
# "gpt-4-turbo" "gpt-4o" "gpt-3.5-turbo" 
selected_style = st.sidebar.radio("Chatbot answer style:", list(style_settings.keys()))

# 4. Filter by Score
score_options = {
    "All": df,
    "1": df[df['Score'] == 1],
    "2": df[df['Score'] == 2],
    "1 or 2": df[df['Score'].isin([1, 2])],
    "3": df[df['Score'] == 3],
    "4": df[df['Score'] == 4],
    "5": df[df['Score'] == 5],
    "4 or 5": df[df['Score'].isin([4, 5])]
}
selected_score = st.sidebar.selectbox("Select Sentiment Range", list(score_options.keys()))

if selected_score != "All":
    df = score_options[selected_score]

# 5. Display Article Summaries
display_text = st.sidebar.radio("Show article summaries?", ("Yes", "No"))

# Debugging output for testing
# st.write(f"Selected Chatbot Style: {selected_style}")
# st.write(f"Selected Publication: {selected_publication}")
# st.write(f"Selected Topic: {selected_topic}")
# st.write(f"Selected Sentiment Range: {selected_score}")
# st.write(f"Show Article Summaries: {display_text}")


#### MAIN PAGE TITLE #########################################################################
# Display the title of the main page
# st.title("AI Media Newsfeed January-June 2024 beta")

# Display the title of the main page using custom HTML for better control over styling
st.markdown('<h2 class="custom-title">AI Media Newsfeed January-June 2024 beta</h2>', unsafe_allow_html=True)

st.info("""This newsfeed presents a curated selection of significant AI media stories with AI-generated summaries and sentiment 
        scores. You can query and converse with the articles using the chatbot prompt.
""")


#### CUSTOM CSS #############################################################################
# Custom CSS to adjust the sidebar width and main content padding
st.markdown("""
    <style>
    /* Adjust the sidebar width */
    [data-testid="stSidebar"] {
        min-width: 220px;
        max-width: 220px;
    }
    /* Adjust the main content to account for the narrower sidebar */
    div.block-container {
        padding-left: 150px;
        margin-left: -250px;
        max-width: 100%;
    }
    /* Add padding to the Plotly chart to avoid overlap with the chatbot */
    .plotly-chart {
        padding-right: 50px;
    }
    /* Custom title styling */
    .custom-title {
        text-align: center;
        margin-top: -50px; /* Adjust this value to reduce white space above */
    }
    </style>
    """, unsafe_allow_html=True)



#### COLUMN WIDTH #############################################################################
# Create a two-column layout with fixed widths
col1, col2 = st.columns([5, 4], gap="large")




#### SET UP COLUMN 1 #############################################################################

with col1:
    #### SET UP CHART ################################################################################
    # Create a 'Month' column to group data by month if it doesn't already exist
    if 'Month' not in df.columns:
        df['Month'] = df['Date'].dt.to_period('M').apply(lambda r: r.start_time)

    # Aggregate data by month
    monthly_summary = df.groupby('Month').apply(lambda x: pd.Series({
        'Percent_4_5': (x['Score'].isin([4, 5]).sum() / len(x)) * 100,
        'Percent_1_2': (x['Score'].isin([1, 2]).sum() / len(x)) * 100,
        'Average_Score': x['Score'].mean()
    })).reset_index()

    # Create traces for the bar chart and line chart with secondary y-axis
    trace1 = go.Bar(
        x=monthly_summary['Month'],
        y=monthly_summary['Percent_4_5'],
        name='4 or 5 Scores',
        marker_color='green',
        yaxis='y1'
    )

    trace2 = go.Bar(
        x=monthly_summary['Month'],
        y=-monthly_summary['Percent_1_2'],  # Negative values for plotting below the x-axis
        name='1 or 2 Scores',
        marker_color='orange',
        yaxis='y1'
    )

    trace3 = go.Scatter(
        x=monthly_summary['Month'],
        y=monthly_summary['Average_Score'],
        name='Average Score',
        mode='lines+markers',
        line=dict(color='black'),
        yaxis='y2'
    )

    # Combine the traces into a figure
    fig = go.Figure(data=[trace1, trace2, trace3])

    # Update the layout to include the secondary y-axis, adjust legend position, and ensure proper x-axis labels
    fig.update_layout(
        title=f'Monthly Sentiment Analysis - {selected_publication}',
        yaxis=dict(
            title='% of Media Stories',
            range=[-70, 80],
            tickfont=dict(
                family='Arial, sans-serif',
                size=12,
                color='black',
                weight='bold'
            ),
            titlefont=dict(
                family='Arial, sans-serif',
                size=16,
                color='black',
                weight='bold'
            )
        ),
        yaxis2=dict(
            title='Average Score',
            overlaying='y',
            side='right',
            range=[1, 5],
            tickvals=[1, 2, 3, 4, 5],
            tickfont=dict(
                family='Arial, sans-serif',
                size=12,
                color='black',
                weight='bold'
            ),
            titlefont=dict(
                family='Arial, sans-serif',
                size=14,
                color='black',
                weight='bold'
            )
        ),
        barmode='overlay',
        xaxis=dict(
            title='Month',
            tickangle=-35,  # Slant the labels
            tickmode='array',  # Use array mode for specific tick values
            tickvals=monthly_summary['Month'],
            tickfont=dict(
                family='Arial, sans-serif',
                size=12,
                color='black',
                weight='bold'
            ),
            titlefont=dict(
                family='Arial, sans-serif',
                size=14,
                color='black',
                weight='bold'
            )
        ),
        legend=dict(
            x=1,
            xanchor='right',
            y=1.2,  # Adjust this value to move the legend higher
            yanchor='top',
            traceorder='normal'
        )
    )

    # Display the Plotly chart above the scrollable text box
    st.plotly_chart(fig, use_container_width=True)
    
    
    #### SCROLLABLE CONTAINER WITH ARTICLES ############################################################################
    
    # Sort the DataFrame by 'Date' in descending order
    df_sorted = df.sort_values(by='Date', ascending=False)
    
    # Generate the HTML content for the articles
    article_html_list = []
    for index, row in df_sorted.iterrows():
        text_html = f"<p style='margin: 5px 0;'>{row['Summary']}</p>" if display_text == "Yes" else ""
        formatted_date = row['Date'].strftime('%B {day}, %Y').format(day=row['Date'].day)
        article_html = f"""
        <div style="margin-bottom: 10px;">
            <h4 style="margin: 5px 0;"><a href="{row['URL']}" target="_blank">{row['Title']}</a></h4>
            <p style="margin: 5px 0;">{row['Publication']} | {formatted_date} | <strong>Score:</strong> {int(row['Score'])}</p>
            {text_html}
            <hr style="margin: 5px 0;">
        </div>
        """
        article_html_list.append(article_html)
    html_content = ''.join(article_html_list)

    # Use the components.html function to render the HTML content within a scrollable div
    components.html(f"""
    <div class="scroll-container" style="height:1400px; overflow-y:auto; padding:20px; border:1px solid #ccc; border-radius:10px; background-color:#f9f9f9;">
        {html_content}
    </div>
    """, height=1500)



#### SET UP COLUMN 2 #############################################################################

with col2:
    
    ############# HEADLINES BOX ################################################################################
   
    headlines = """
    <h2>How did top tier media perceive Microsoft's role in AI during January-June 2024?</h2>
    <p>You are a helpful expert with deep technical and policy knowledge about the full range of Microsoft's 
    environmental sustainability policies and programs, especially those that concern Microsoft's
    commitment to become carbon negative, water positive, zero waste, and to protect more land than
    it uses.</p>
    <p>Your job is to provide maximally complete, detailed, logical, clear, and accurate answers to questions 
    about:</p>
    <ol>
        <li>Microsoft's environmental sustainability commitments and the policies and programs Microsoft is
        using to fulfill those commitments.</li>
        <li>The technologies and economic techniques Microsoft is applying or developing in pursuit of those commitments.</li>
        <li>The government and corporate policies that Microsoft recommends that governments and corporations
        should adopt in order to help the world achieve zero carbon and a truly sustainable environment.</li>
    </ol>
    <p>You always pay close attention to the exact phrasing of the user's question and you always 
    deliver an answer that matches every specific detail of the user's expressed intention.</p>
    <p>You always give the fullest and most complete answers possible in substantial paragraphs 
    full of relevant details, concrete examples, and logical analysis.</p>
    <p>You never make up facts. You are not lazy and you do not skimp on important context.
    You always write in very polished and clear business prose, such as might be published
    in a leading business periodical like Harvard Business Review.</p>
    <p>The users of this chatbot are making critical business decisions based on 
    your answers, and they need the highest quality answers possible that comply to the letter
    with their precise instructions.</p>
    <p>Finally, we have observed in your recent answers a distinct tendency to be lazy by providing
    answers that are short, lacking in extensive detail, and lacking in substantive analysis. It is 
    imperative that in this answer you overcome your instinct toward laziness and that you provide
    the richest, most detailed, most extensive answer that is possible to assemble from your
    sources, while still scrupulously respecting user instructions regarding the length of your response.
    YOU NEVER GIVE SHORT ANSWERS UNLESS SPECIFICALLY INSTRUCTED TO DO SO. In most cases a one
    paragraph answer is TOO SHORT.</p>
"""

    # Apply custom CSS to ensure minimal spacing and restore the container
    st.markdown("""
        <style>
        .scroll-container {
            height: 200px;
            overflow-y: auto;
            padding: 20px;
            border: 1px solid #ccc;
            border-radius: 10px;
            background-color: #f9f9f9;
        }
        .scroll-container h2 {
            margin-top: 0;
            margin-bottom: 10px;
        }
        .scroll-container p, .scroll-container ol, .scroll-container li {
            margin: 5px 0;
        }
        </style>
    """, unsafe_allow_html=True)

#### SCROLLABLE TEXT BOX WITH HEADLINES TEXT ##################################################

# Render the formatted text with a centered and line-broken headline in a scrollable div above the chatbot
    components.html(f"""
    <style>
        .centered-headline {{
            text-align: center;
            display: block;
        }}
        .centered-headline span {{
            display: block;
        }}
    </style>
    <div class="scroll-container" style="height:250px; overflow-y:auto; padding:20px; border:1px solid #ccc; border-radius:10px; background-color:#f9f9f9;">
        <h2 class="centered-headline">
            <span>Media Perception of Microsoft in AI</span>
            <span>January-June 2024</span>
        </h2>
        <ol>
            <li><strong>Leadership in Ethical AI Practices:</strong> Microsoft is often portrayed as a leader in ethical AI development, which is a positive perception that can enhance its brand reputation. For instance, Politico has highlighted Microsoft's proactive stance in forming AI ethics guidelines and its involvement in initiatives like the AI Election Accords, which aim to prevent the misuse of AI in elections. This positions Microsoft as a responsible leader in AI, contrasting with competitors who may be seen as less committed to ethical considerations.</li>
            <li><strong>Dominance in AI Infrastructure:</strong> The Financial Times and Wall Street Journal have noted Microsoft's significant investments in AI, particularly through its partnership with OpenAI and the development of Azure as a leading cloud platform for AI solutions. This is perceived positively as it showcases Microsoft's commitment to advancing AI technology. However, this dominance also brings scrutiny regarding competitive practices and potential monopolistic behavior, which could lead to antitrust concerns.</li>
            <li><strong>Innovation vs. Regulation:</strong> There is a nuanced view presented by these media outlets, particularly the New York Times, regarding the balance Microsoft seeks between innovation and regulation. While the company advocates for reasonable regulatory frameworks to ensure AI safety and ethics, there is also concern about over-regulation stifling innovation. Microsoft's executives need to be aware of the ongoing debate and be prepared to engage in discussions that advocate for balanced policies that do not hinder technological advancement.</li>
            <li><strong>Impact on Society and Economy:</strong> Articles from these publications often discuss the broader impact of Microsoft's AI on society and the economy. For example, the integration of AI into products like Bing and Office is seen as a move that could potentially reshape industries and alter the labor market. While this is often viewed as a positive driver of efficiency and growth, there is also coverage of potential negative impacts, such as job displacement and privacy concerns. Microsoft should continue to address these issues transparently and promote the benefits of AI in enhancing human capabilities rather than replacing them.</li>
            <li><strong>Handling of AI-generated Data and Content:</strong> Concerns about the handling of data and the generation of content using AI technologies like ChatGPT have been a focus in reports by Politico and the New York Times. The legal challenges and copyright issues surrounding the use of AI to generate or modify content are areas of potential vulnerability for Microsoft. It is crucial for the executives to monitor these discussions closely and develop strategies that ensure compliance and respect for intellectual property rights, while also advocating for laws that support innovation in AI content generation.</li>
        </ol>
    </div>
    """, height=300)



    #### CHATBOT SETUP ################################################################################
    # Place your chatbot code here

    # Get the selected system prompt and model
    system_prompt = style_settings[selected_style]["s_prompt"]
    model = style_settings[selected_style]["model"]
    # enable logging of the selected radio button
    prompt_type = selected_style

    # Get the selected system prompt and model
    # system_prompt = depth_prompt
    # model = "gpt-4-turbo"  # "gpt-3.5-turbo" "gpt-4-turbo" "gpt-4o"

    # SET UP LLM AND PARAMETERS
    openai.api_key = st.secrets.openai_key

    llm = OpenAI(
        api_key=openai.api_key,
        model=model,
        temperature=0.5,
        system_prompt=system_prompt
    )

    # SET UP LLAMAINDEX SETTINGS
    chunk_size = 512
    chunk_overlap = 256
    chunk_params = (chunk_size, chunk_overlap)
    Settings.llm = llm
    Settings.chunk_size = chunk_size
    Settings.chunk_overlap = chunk_overlap

    if "messages" not in st.session_state.keys():  # Initialize the chat messages history
        st.session_state.messages = [
            {"role": "assistant", "content": """Ask a question about trends, opinions, or facts covered
             in the media articles. 
             """}
        ]

    # LOAD DOCS INTO LLAMAINDEX, CREATE INDEX (AND RELOAD IF IT ALREADY EXISTS)
    # persist_dir = './index'
    @st.cache_resource(show_spinner=False)
    def load_data():
        with st.spinner(text="Loading the reports. This will take a few minutes."):
            # Define the path to the index file
            # persist_dir = persist_dir
            # persist_dir = './index_deepfakes'
            persist_dir = './index'

            # Check if the index directory exists, create if not
            if not os.path.exists(persist_dir):
                os.makedirs(persist_dir)

            # Now attempt to load or create the index
            try:
                storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
                index = load_index_from_storage(storage_context)
            except FileNotFoundError:
                # Index not found, create it
                input_dir = r"C:\Users\Jeff\Dropbox\Jeff DropBox\Cloud Assurance\LLM Projects\Newsfeed projects\Newsfeed docs\AI reg docs Jan-May 2024\AI PDF deepfakes"
                # input_dir = r"C:\Users\Jeff\Dropbox\Jeff DropBox\Cloud Assurance\LLM Projects\Newsfeed projects\Newsfeed docs\AI reg docs Jan-May 2024\AI pdfs done"
                reader = SimpleDirectoryReader(input_dir)
                docs = reader.load_data()
                index = VectorStoreIndex.from_documents(docs)

                # Save the index to the file
                index.storage_context.persist(persist_dir=persist_dir)

            return index

    index = load_data()

    # DEFINE THE RUN_CHATS FUNCTION
    def run_chats(query):

        search_time = 0.0

        similarity_top_k = 40

        start_time_search = time.time()  # time vector search
        chat_engine = index.as_chat_engine(chat_mode="condense_question",
                                           streaming=True,
                                           similarity_top_k=similarity_top_k,
                                           )  # verbose=True # streaming=True
        end_time_search = time.time()

        result = chat_engine.chat(query)  # chat_engine.stream_chat(query)
        # Calculate search time
        search_time = end_time_search - start_time_search

        # Store the values of k
        query_params = similarity_top_k
        # result.print_response_stream()

        return result, query_params, search_time



#### CREATE CHAT ENGINE #############################################################################

    # Initialize the chat engine
    if "chat_engine" not in st.session_state:
        st.session_state.chat_engine = None

    # Initialize the messages history if not present
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask a question and use the buttons on the left to tell the chatbot what kind of answer you want:"}
        ]

    # Initialize the input key if not present
    if 'input_key' not in st.session_state:
        st.session_state.input_key = 0

    # Function to handle user input and generate a response
    def handle_user_input(prompt):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                start_time = time.time()
                response, query_params, search_time = run_chats(prompt)
                response_time = time.time() - start_time
                st.write(response.response)
                message = {"role": "assistant", "content": response.response}
                st.session_state.messages.append(message)  # Add response to message history

        # Set the processing state to False after response
        st.session_state.is_processing = False
        # Increment the input key to ensure a unique key for the next input box
        st.session_state.input_key += 1

    # Initialize processing state if not present
    if 'is_processing' not in st.session_state:
        st.session_state.is_processing = False

    # Function to clear the chat history
    def clear_chat():
        st.session_state.messages = [
            {"role": "assistant", "content": "Ask a question and use the buttons on the left to tell the chatbot what kind of answer you want:"}
        ]
        st.session_state.input_key += 1
        st.rerun()

    # Display the prior chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Apply custom CSS to style the button and align the form
    st.markdown("""
        <style>
        div[data-testid="stForm"] button {
            width: 100px !important; /* Set the button width explicitly */
            height: 40px !important;
            padding: 5px 10px !important;
            font-size: 12px !important; 
            line-height: 16px !important; /* Ensure this is less than the height to center the text */
            text-align: center !important;
            vertical-align: middle !important;
            border: 1px solid #ccc !important;
            border-radius: 10px !important;
            background-color: #f9f9f9 !important;
            cursor: pointer !important;
        }
        div[data-testid="stForm"] button:hover {
            background-color: #e6e6e6 !important;
        }
        /* Align the form vertically with the text input box */
        div[data-testid="stForm"] {
            display: flex !important;
            align-items: center !important;
            height: 40px !important; /* Set the form height explicitly */
            justify-content: center !important; /* Center the form content */
            margin-top: 1px !important; /* Adjust margin-top if necessary to align with the input box */
            border: none !important; /* Remove the border */
            width: 100px !important; /* Set the form width explicitly */
        }
        </style>
        """, unsafe_allow_html=True)

    # Layout with input box and custom HTML button
    col1, col2 = st.columns([4, 1])
    with col1:
        if not st.session_state.is_processing:
            prompt = st.chat_input("Your question:", key=f"user_input_{st.session_state.input_key}")
            if prompt:
                st.session_state.is_processing = True
                handle_user_input(prompt)
                st.rerun()

    with col2:
        with st.form(key='erase_form'):
            submitted = st.form_submit_button("Erase history")
            if submitted:
                clear_chat()

