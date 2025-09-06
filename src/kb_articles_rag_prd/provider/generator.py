from kb_articles_rag_prd.core.generator import BaseGenerator
from langchain_core.messages import BaseMessage
from langchain_core.documents import Document
from langchain.prompts import load_prompt


class KBArticlesTextGenerator(BaseGenerator):
    """Basic implementation

    Args:
        BaseGenerator (KBArticlesTextGenerator): 
    """

    def get_context(self, docs: list[Document]) -> str:
        """This is used to format docs
        """
        out = []
        for d in docs:
            src  = d.metadata.get("source", "unknown")
            page = d.metadata.get("page")
            tag  = f"{src}" + (f" p.{page}" if page is not None else "")
            txt  = d.page_content.strip().replace("\n", " ")
            out.append(f"[source: {tag}] {txt}")
        return "\n\n".join(out[:12])

    def respond(self, question) -> BaseMessage:
        retrieved_docs = self._vector_store.as_retriever().invoke(question)
        context = self.get_context(retrieved_docs)
        prompt_template = load_prompt('prompts/kbarticles_employee.yml')
        formatted_prompt = prompt_template.format(context=context, question=question)
        # get the response from the model
        result = self._model.invoke(formatted_prompt)
        return result