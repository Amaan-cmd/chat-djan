# chat/callbacks.py
from langchain_core.callbacks import BaseCallbackHandler

class MyCustomCallbackHandler(BaseCallbackHandler):
    def on_chain_start(self, serialized, inputs, **kwargs):
        print("Chain started with inputs:", inputs)

    def on_chain_end(self, outputs, **kwargs):
        print("Chain ended with outputs:", outputs)

    def on_chain_error(self, error, **kwargs):
        print("Chain error:", error)