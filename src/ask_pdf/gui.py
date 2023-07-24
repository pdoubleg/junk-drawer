__version__ = "0.0.1"
app_name = "LibertyGPT Demo"

from typing import List
import streamlit as st
st.set_page_config(layout='wide', page_title=f'{app_name} {__version__}')
ss = st.session_state
if 'debug' not in ss: ss['debug'] = {}
import css
st.write(f'<style>{css.v1}</style>', unsafe_allow_html=True)


import prompts
import model

import time

def ui_spacer(n=2, line=False, next_n=0):
	for _ in range(n):
		st.write('')
	if line:
		st.tabs([' '])
	for _ in range(next_n):
		st.write('')
  

def wrap_text_in_html(text: List[str]) -> str:
    """Wraps each text block separated by newlines in <p> tags"""
    if isinstance(text, list):
        # Add horizontal rules between pages
        text = "\n<hr/>\n".join(text)
    return "".join([f"<p>{line}</p>" for line in text.split("\n")])
  

def ui_info():
	st.markdown(f"""
	# LibertyGPT Demo
	version {__version__}
	
	Question answering over (public) documents built on top of LibertyGPT.
	""")
	ui_spacer(1)
	st.write("Made by [Paul Gargano](paul.gargano@libertymutual.com).", unsafe_allow_html=True)
	ui_spacer(1)
	st.markdown("""
		Please note this app is for demonstration purposes only.
		""")


def ui_task_template():
	st.selectbox('task prompt template', prompts.TASK.keys(), key='task_name')

def ui_task():
	x = ss['task_name']
	st.text_area('task prompt', prompts.TASK[x], key='task')


def ui_output():
	output = ss.get('output','')
	st.markdown(output)
 
def ui_source_output():
	source_output = ss.get('source','')
	# source_html = wrap_text_in_html(source_output)
	st.markdown(source_output, unsafe_allow_html=True)

def b_clear():
	if st.button('clear output'):
		ss['output'] = ''
		ss['source'] = ''

def b_reload():
	if st.button('reload prompts'):
		import importlib
		importlib.reload(prompts)


def output_add(q,a):
	if 'output' not in ss: ss['output'] = ''
	q = q.replace('$',r'\$')
	a = a.replace('$',r'\$')
	new = f'#### {q}\n{a}\n\n'
	ss['output'] = new + ss['output']
 
def output_source(s):
	if 'source' not in ss: ss['source'] = ''
	s = s.replace('$',r'\$')
	new = f'#### {s}\n\n'
	ss['source'] = new + ss['source']


with st.sidebar:
	ui_info()
	ui_spacer(2)
	with st.expander('advanced'):
		b_clear()
		b_reload()
		ui_task_template()
		ui_task()
  
question = st.chat_input(placeholder='Enter question here', key='question_', disabled=False, on_submit=None)
if question:
	question = f"{question}"
	ss['question'] = question
	resp = model.query(question)	
	q = question.strip()
	a = str(resp)
	s = resp.source_nodes[0].node.get_text()
	ss['answer'] = a
	ss['source'] = s
	output_add(q, a)

col1, col2 = st.columns(2)
with col1:
    ui_output()

with col2:
    with st.container():
        ui_source_output()
    
