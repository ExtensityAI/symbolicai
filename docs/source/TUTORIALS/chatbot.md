# Writing a Custom Chatbot

---

## ⚠️  Outdated or Deprecated Documentation ⚠️
This documentation is outdated and may not reflect the current state of the SymbolicAI library. This page might be revived or deleted entirely as we continue our development. We recommend using more modern tools that infer the documentation from the code itself, such as [DeepWiki](https://deepwiki.com/ExtensityAI/symbolicai). This will ensure you have the most accurate and up-to-date information and give you a better picture of the current state of the library.

---

Click here for [interactive notebook](https://github.com/ExtensityAI/symbolicai/blob/main/notebooks/ChatBot.ipynb)

```python
import os
import warnings
warnings.filterwarnings('ignore')
os.chdir('../') # set the working directory to the root of the project
from symai import *
from symai.components import *
from IPython.display import display
```

Writing a chatbot is fairly easy with our framework. All we need to do is basically derive from the ChatBot class and implement the forward method. The base class ChatBot has already some helper capabilities and context detection dictionaries. All we have to do is use the self.narrate method to instruct our chatbot to say what we want.

Afterwards, we can use the self.context_choice method to classify the context of the user input. This is done by using a dictionary of context keywords. The self.context_choice method returns the context key that matches the user input. This key can then be used to determine the next action / condition of the chatbot.

By creating an instance of the SymbiaChat and calling the forward method, we can start a chat with our chatbot. The forward method takes a user input and returns a chatbot response.

See the following example:

```python
from symai.chat import ChatBot
from symai.interfaces import Interface


class SymbiaChat(ChatBot):
    def forward(self) -> str:
        message = self.narrate('Symbia introduces herself, writes a greeting message and asks how to help.')
        while True:
            # query user
            usr = self.input(message)

            # detect context
            ctxt = self.context_choice(usr)

            if 'option 3' in ctxt: # exit
                self.narrate('Symbia writes goodbye message.', end=True)
                break # end chat

            elif 'option 4' in ctxt: # help
                message = self.narrate('Symbia writes for each capability one sentence.',
                                       context=self.capabilities)

            elif 'option 1' in ctxt: # chit chat
                message = self.narrate('Symbia replies to the user question in a casual way.')

            elif 'option 2' in ctxt:
                # detect command
                option = self.capabilities_choice(usr)

                if 'option 1' in option:
                    q = usr.extract('user query request')
                    rsp = self.search(q)
                    message = self.narrate('Symbia replies to the user based on the online search results.',
                                           context=rsp)
                elif 'option 2' in option:
                    q = usr.extract('URL')
                    site = self.crawler(q)
                    site.save('tmp.html')
                    message = self.narrate('Symbia explains that the website is downloaded to the `tmp.html` file.')
                elif 'option 3' in option:
                    pass

                # TODO ...
```

```python
from symai.chat import SymbiaChat

chat = SymbiaChat()
chat()
```

The implemented chatbot can answer trivia questions, use the Google search engine to retrieve information, and download and save web pages.
