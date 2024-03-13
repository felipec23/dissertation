import streamlit as st

# Create state variable 
if 'search_engine' not in st.session_state:
    st.session_state.search_engine = "By papers"

def sidebar_code():
    with st.sidebar:

        # Sidebar title
        st.title("Search Engine")



        # Radio button to select the search engine
        search_engine = st.radio("Select search type", ("By papers", "By variables"))

        # Save the search engine in the state variable
        st.session_state.search_engine = search_engine

        # Create a button to show the feedback form

        # with st.form(key='my_form_2', clear_on_submit=True):

        #     # st.title("Feedback")
        #     st.markdown("Please, let us know if you have any feedback or suggestions for improvement.")

        #     # Create a text area to get the feedback
        #     feedback = st.text_area("Feedback", value="", height=100, key="feedback")
        #     submit_button = st.form_submit_button(label='Submit')
        #     placeholer_feedback = st.empty()

        #     if submit_button:
        #         print("Feedback: ", feedback)
        #         placeholer_feedback.success("Feedback sent, thanks!")