import * as dotenv from "dotenv";
import { CustomPDFLoader } from "../utils/CustomPDFLoader";
import { VectorDBQAChain } from "langchain/chains";
import { OpenAIEmbeddings } from "langchain/embeddings";
import { OpenAI } from "langchain/llms/OpenAI";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { HNSWLib } from "langchain/vectorstores";

// Load environment variables from .env file
dotenv.config();

export const main = async () => {
  try {
    const model = new OpenAI({ maxTokens: 1000, temperature: 0.1 });

    const loader = new CustomPDFLoader("swiggy.pdf");
    const doc = await loader.load();

    const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000 });
    const docs = await textSplitter.splitDocuments(doc);

    const vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());
    const qaChain = VectorDBQAChain.fromLLM(model, vectorStore);

    // Assuming there is only one question to prevent multiple calls.
    const question = "Convert the invoice of purchase as json";
    const payload = {
      input_documents: docs,
      query: "You only reply as json" + question,
    };

    console.log("Sending request to OpenAI with payload:", payload);
    const answer = await qaChain.call(payload);
    console.log("Received answer from OpenAI:", answer);
    console.log("\n\n> " + question + "\n" + answer.text);

  } catch (e) {
    console.error("An error occurred:", e);
  }
};

main();
