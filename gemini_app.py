import streamlit as st
import anthropic
import google.generativeai as genai
from typing import Dict, List
import json
import PyPDF2
import io
from pathlib import Path
import base64

class DocumentProcessor:
    def __init__(self, api_key: str, provider: str):
        self.provider = provider
        if provider == "claude":
            self.client = anthropic.Anthropic(api_key=api_key)
        elif provider == "gemini":
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-pro')
    
    def extract_text_from_pdf(self, uploaded_file) -> str:
        try:
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error reading PDF: {e}")
            return ""
    
    def extract_signing_rules_smart(self, document_content: str) -> Dict:
        prompt = f"""
        Analyze this document and automatically detect all sections containing signing rules or approval checkboxes.
        
        Task:
        1. Scan entire document for sections with signing rules/approval checkboxes
        2. Identify checkbox elements: □, ☐, ■, ☑, ✓, X, numbers in brackets, parentheses
        3. Determine approval status: filled = approved, empty = not approved
        4. Extract complete rule text
        
        Document:
        {document_content}
        
        Return JSON object:
        {{
            "status": "success" or "error",
            "message": "description of findings",
            "sections_found": [
                {{
                    "section_name": "section name",
                    "section_number": "section number",
                    "location": "location in document"
                }}
            ],
            "total_rules": number,
            "approved_count": number,
            "approved_rules": [
                {{
                    "rule_number": number,
                    "rule_text": "rule text without checkbox",
                    "checkbox_content": "checkbox content",
                    "section": "section name",
                    "is_approved": true
                }}
            ],
            "all_rules": [
                {{
                    "rule_number": number,
                    "rule_text": "rule text without checkbox",
                    "checkbox_content": "checkbox content", 
                    "section": "section name",
                    "is_approved": true/false
                }}
            ]
        }}
        
        Checkbox detection rules:
        - Approved: ✓, X, ☑, ■, numbers, letters, symbols
        - Not approved: □, ☐, ( ), [ ], empty spaces
        
        Return only JSON, no other text.
        """
        
        try:
            if self.provider == "claude":
                message = self.client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=4000,
                    messages=[{"role": "user", "content": prompt}]
                )
                response_text = message.content[0].text.strip()
            else:
                response = self.model.generate_content(prompt)
                response_text = response.text.strip()
            
            if response_text.startswith('```json'):
                response_text = response_text[7:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:-3]
            
            return json.loads(response_text)
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Error processing document: {str(e)}",
                "sections_found": [],
                "total_rules": 0,
                "approved_count": 0,
                "approved_rules": [],
                "all_rules": []
            }

def display_pdf_pages(pdf_file):
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        total_pages = len(pdf_reader.pages)
        
        if 'current_page' not in st.session_state:
            st.session_state.current_page = 0
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            if st.button("◀ Previous", disabled=st.session_state.current_page == 0):
                st.session_state.current_page -= 1
                st.rerun()
        
        with col2:
            st.write(f"Page {st.session_state.current_page + 1} of {total_pages}")
        
        with col3:
            if st.button("Next ▶", disabled=st.session_state.current_page >= total_pages - 1):
                st.session_state.current_page += 1
                st.rerun()
        
        current_page_text = pdf_reader.pages[st.session_state.current_page].extract_text()
        
        st.text_area("Document Content (Current Page):", value=current_page_text, height=400, disabled=True)
        
        return total_pages
        
    except Exception as e:
        st.error(f"Error displaying PDF: {e}")
        return 0

def display_signing_rules(results: Dict):
    if results["status"] == "error":
        st.error(f"Error: {results['message']}")
        return
    
    st.success(f"✅ Successfully processed document!")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Rules", results["total_rules"])
    with col2:
        st.metric("Approved Rules", results["approved_count"])
    with col3:
        st.metric("Sections Found", len(results["sections_found"]))
    
    if results["sections_found"]:
        st.subheader("📍 Sections Found")
        for section in results["sections_found"]:
            st.write(f"• **{section['section_name']}** {section.get('section_number', '')}")
    
    if results["approved_rules"]:
        st.subheader("✅ Approved Signing Rules")
        for rule in results["approved_rules"]:
            with st.expander(f"Rule {rule['rule_number']}: {rule['rule_text'][:50]}..."):
                st.write(f"**Section:** {rule.get('section', 'N/A')}")
                st.write(f"**Checkbox Content:** `{rule['checkbox_content']}`")
                st.write(f"**Full Rule Text:** {rule['rule_text']}")
                st.success("✅ APPROVED")
    
    with st.expander("📋 All Rules Summary"):
        for rule in results["all_rules"]:
            status_icon = "✅" if rule["is_approved"] else "❌"
            status_text = "APPROVED" if rule["is_approved"] else "NOT APPROVED"
            
            st.write(f"{status_icon} **Rule {rule['rule_number']}:** {rule['rule_text']}")
            st.write(f"   📍 Section: {rule.get('section', 'N/A')} | Checkbox: `{rule['checkbox_content']}` | Status: {status_text}")
            st.divider()

def main():
    st.set_page_config(
        page_title="Document Signing Rules Extractor",
        page_icon="📄",
        layout="wide"
    )
    
    st.title("📄 Document Signing Rules Extractor")
    st.markdown("Upload a PDF document to automatically extract and analyze signing rules with checkboxes")
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        provider = st.selectbox("Select AI Provider:", ["claude", "gemini"])
        
        if provider == "claude":
            api_key = st.text_input("Enter Claude API Key:", type="password")
            st.info("Get your API key from: https://console.anthropic.com/")
        else:
            api_key = st.text_input("Enter Gemini API Key:", type="password")
            st.info("Get your API key from: https://makersuite.google.com/app/apikey")
        
        st.markdown("---")
        st.markdown("### 📋 How it works:")
        st.markdown("""
        1. Upload a PDF document
        2. Navigate through pages
        3. Click 'Extract Rules' to analyze
        4. View approved signing rules
        """)
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        st.header("📖 Document Viewer")
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            st.subheader("Document Pages")
            total_pages = display_pdf_pages(uploaded_file)
        
        with col2:
            st.subheader("Document Info")
            st.write(f"**Filename:** {uploaded_file.name}")
            st.write(f"**File size:** {uploaded_file.size:,} bytes")
            st.write(f"**Total pages:** {total_pages}")
        
        st.markdown("---")
        
        if st.button("🔍 Extract Signing Rules", type="primary", use_container_width=True):
            if not api_key:
                st.error("Please enter your API key in the sidebar!")
                return
            
            with st.spinner("Analyzing document for signing rules..."):
                try:
                    processor = DocumentProcessor(api_key, provider)
                    uploaded_file.seek(0)
                    document_text = processor.extract_text_from_pdf(uploaded_file)
                    
                    if document_text:
                        results = processor.extract_signing_rules_smart(document_text)
                        st.session_state.extraction_results = results
                    else:
                        st.error("Could not extract text from PDF")
                        
                except Exception as e:
                    st.error(f"Error processing document: {e}")
        
        if 'extraction_results' in st.session_state:
            st.header("🎯 Extraction Results")
            display_signing_rules(st.session_state.extraction_results)
    
    else:
        st.info("👆 Please upload a PDF document to get started")
        
        st.markdown("---")
        st.markdown("### 🔍 What this app detects:")
        st.markdown("""
        - **Checkbox formats:** ☑, ✓, X, ■, [X], (X), numbered boxes
        - **Empty checkboxes:** □, ☐, [ ], ( )
        - **Signing rules** across all document sections
        - **Approval status** based on checkbox content
        """)

if __name__ == "__main__":
    main()
