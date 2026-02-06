{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "921f72c8-3fec-4d05-9930-972061c6b2e0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Collecting fastapi\n",
      "  Downloading fastapi-0.128.2-py3-none-any.whl.metadata (30 kB)\n",
      "Collecting starlette<0.51.0,>=0.40.0 (from fastapi)\n",
      "  Downloading starlette-0.50.0-py3-none-any.whl.metadata (6.3 kB)\n",
      "Requirement already satisfied: pydantic>=2.7.0 in f:\\softwares\\anaconda\\lib\\site-packages (from fastapi) (2.10.3)\n",
      "Requirement already satisfied: typing-extensions>=4.8.0 in f:\\softwares\\anaconda\\lib\\site-packages (from fastapi) (4.12.2)\n",
      "Collecting typing-inspection>=0.4.2 (from fastapi)\n",
      "  Downloading typing_inspection-0.4.2-py3-none-any.whl.metadata (2.6 kB)\n",
      "Collecting annotated-doc>=0.0.2 (from fastapi)\n",
      "  Downloading annotated_doc-0.0.4-py3-none-any.whl.metadata (6.6 kB)\n",
      "Requirement already satisfied: anyio<5,>=3.6.2 in f:\\softwares\\anaconda\\lib\\site-packages (from starlette<0.51.0,>=0.40.0->fastapi) (4.7.0)\n",
      "Requirement already satisfied: idna>=2.8 in f:\\softwares\\anaconda\\lib\\site-packages (from anyio<5,>=3.6.2->starlette<0.51.0,>=0.40.0->fastapi) (3.7)\n",
      "Requirement already satisfied: sniffio>=1.1 in f:\\softwares\\anaconda\\lib\\site-packages (from anyio<5,>=3.6.2->starlette<0.51.0,>=0.40.0->fastapi) (1.3.0)\n",
      "Requirement already satisfied: annotated-types>=0.6.0 in f:\\softwares\\anaconda\\lib\\site-packages (from pydantic>=2.7.0->fastapi) (0.6.0)\n",
      "Requirement already satisfied: pydantic-core==2.27.1 in f:\\softwares\\anaconda\\lib\\site-packages (from pydantic>=2.7.0->fastapi) (2.27.1)\n",
      "Downloading fastapi-0.128.2-py3-none-any.whl (104 kB)\n",
      "Downloading starlette-0.50.0-py3-none-any.whl (74 kB)\n",
      "Downloading annotated_doc-0.0.4-py3-none-any.whl (5.3 kB)\n",
      "Downloading typing_inspection-0.4.2-py3-none-any.whl (14 kB)\n",
      "Installing collected packages: typing-inspection, annotated-doc, starlette, fastapi\n",
      "\n",
      "   ---------- ----------------------------- 1/4 [annotated-doc]\n",
      "   -------------------- ------------------- 2/4 [starlette]\n",
      "   -------------------- ------------------- 2/4 [starlette]\n",
      "   -------------------- ------------------- 2/4 [starlette]\n",
      "   -------------------- ------------------- 2/4 [starlette]\n",
      "   -------------------- ------------------- 2/4 [starlette]\n",
      "   -------------------- ------------------- 2/4 [starlette]\n",
      "   ------------------------------ --------- 3/4 [fastapi]\n",
      "   ------------------------------ --------- 3/4 [fastapi]\n",
      "   ------------------------------ --------- 3/4 [fastapi]\n",
      "   ------------------------------ --------- 3/4 [fastapi]\n",
      "   ------------------------------ --------- 3/4 [fastapi]\n",
      "   ------------------------------ --------- 3/4 [fastapi]\n",
      "   ------------------------------ --------- 3/4 [fastapi]\n",
      "   ------------------------------ --------- 3/4 [fastapi]\n",
      "   ------------------------------ --------- 3/4 [fastapi]\n",
      "   ------------------------------ --------- 3/4 [fastapi]\n",
      "   ------------------------------ --------- 3/4 [fastapi]\n",
      "   ---------------------------------------- 4/4 [fastapi]\n",
      "\n",
      "Successfully installed annotated-doc-0.0.4 fastapi-0.128.2 starlette-0.50.0 typing-inspection-0.4.2\n"
     ]
    }
   ],
   "source": [
    "!pip install fastapi # To install the fastapi"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "8acc5a5f-7405-4ba4-8e13-a1cac28cc10e",
   "metadata": {},
   "outputs": [],
   "source": [
    "from fastapi import FastAPI #creates the web API\n",
    "import joblib # loads saved ML model\n",
    "import pandas as pd # handles input data in tabular form\n",
    "import numpy as np # reverse log transformation (expm1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "56f7a214-d8e7-4852-b04c-23469d9ce608",
   "metadata": {},
   "outputs": [],
   "source": [
    "app = FastAPI(title=\"House Price Prediction API\")\n",
    "# Creates your API application. app is the object that handles requests. Title appears in API docs"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "d7f451ea-393d-4c7b-b288-cc6eb2089aae",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = joblib.load(\"ridge_house_price_model.pkl\")\n",
    "model_features = joblib.load(\"model_features.pkl\")\n",
    "# This is to load the model and feature list (Column Names)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "328e5578-b926-47cf-a214-d6bcbec0d1aa",
   "metadata": {},
   "outputs": [],
   "source": [
    "@app.get(\"/\")\n",
    "def home():\n",
    "    return {\"message\": \"House Price Prediction API is running\"}\n",
    "\n",
    "# Simple endpoint to verify API is alive\n",
    "# Used by cloud platforms to check health"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "e9479d85-a147-4e51-a94f-3dc65bdefe62",
   "metadata": {},
   "outputs": [],
   "source": [
    "@app.post(\"/predict\")\n",
    "def predict_price(data: dict):\n",
    "    # Convert input to DataFrame\n",
    "    input_df = pd.DataFrame([data])\n",
    "\n",
    "    # Ensure all required features exist, If some values are missing then 0 will be assigned automatically.\n",
    "    for col in model_features:\n",
    "        if col not in input_df.columns:\n",
    "            input_df[col] = 0\n",
    "\n",
    "    # Reorder columns\n",
    "    input_df = input_df[model_features]\n",
    "\n",
    "    # Prediction (log scale)\n",
    "    log_price = model.predict(input_df)[0]\n",
    "\n",
    "    # Convert back to original scale\n",
    "    price = np.expm1(log_price)\n",
    "\n",
    "    return {\"predicted_house_price\": round(price, 2)}"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.13.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
