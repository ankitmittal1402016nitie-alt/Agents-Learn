import streamlit as st

st.title("Streamlit Test App")
st.write("Hello! If you can see this, Streamlit is working correctly!")

# Add a simple interactive element
if st.button("Click me!"):
    st.balloons()