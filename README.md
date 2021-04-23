# Vaccination Center Assistant
A series of applications to help monitor a waiting room, log vaccination events, and monitor a post-vaccination room, and write out
event logs to help inform vaccination center logistics.

## Requirements
- [alwaysAI account](https://alwaysai.co/auth?register=true)
- [alwaysAI CLI tools](https://dashboard.alwaysai.co/docs/getting_started/development_computer_setup.html)

## Running
Run each app individually as outlined [here](https://alwaysai.co/blog/building-and-deploying-apps-on-alwaysai)
Note that for the application in the `waiting` folder, you will have to train and add in your own mask detection model! Access our docs
for information on [training](https://alwaysai.co/docs/model_training/index.html) and [using models](https://alwaysai.co/docs/alwaysai_workflow/working_with_models.html).
Also note you can change the server URL to be whatever you choose for your dashboard. Each application is currently set up to stream
from a web camera (camera 0), but you can choose any file or video stream that fits your use case.

## Support
Docs: https://dashboard.alwaysai.co/docs/getting_started/introduction.html

Community Discord: https://discord.gg/rjDdRPT

Email: support@alwaysai.co