import { promises as fs } from 'fs';
import pdf from 'pdf-parse';
import { Document } from 'langchain/document';
import { PromptTemplate } from "langchain/prompts";
import dotenv from 'dotenv';
import { OpenAI } from 'langchain/llms/openai';

dotenv.config();

// CustomPDFLoader class (as defined earlier)
export abstract class BufferLoader extends BaseDocumentLoader {
  constructor(public filePathOrBlob: string | Blob) {
    super();
  }

  protected abstract parse(raw: Buffer, metadata: Document['metadata']): Promise<Document[]>;

  public async load(): Promise<Document[]> {
    let buffer: Buffer;
    let metadata: Record<string, string>;

    try {
      if (typeof this.filePathOrBlob === 'string') {
        buffer = await fs.readFile(this.filePathOrBlob);
        metadata = { source: this.filePathOrBlob };
      } else {
        buffer = Buffer.from(await this.filePathOrBlob.arrayBuffer());
        metadata = { source: 'blob', blobType: this.filePathOrBlob.type };
      }
      return this.parse(buffer, metadata);
    } catch (error) {
      console.error("Error loading document:", error);
      throw error;
    }
  }
}

export class CustomPDFLoader extends BufferLoader {
  public async parse(raw: Buffer, metadata: Document['metadata']): Promise<Document[]> {
    const parsed = await pdf(raw);
    return [new Document({ pageContent: parsed.text, metadata })];
  }
}

// Functions for LangChain processing
async function readPdfAsDocument(filePath: string) {
    const loader = new (filePath);
    return await loader.load();
}

function initializeLangChain() {
    return new OpenAI({
        openAIApiKey: process.env.OPENAI_API_KEY,
    });
}

function createPromptTemplate() {
    return PromptTemplate.fromTemplate("This is an invoice of a purchase, parse this as a JSON file. You only reply back JSON");
}

async function processDocument(document: Document[], langChainModel: OpenAI) {
    const prompt = createPromptTemplate();
    const formattedPrompt = await prompt.format({ documentContent: document[0].pageContent });
    return await langChainModel.in(formattedPrompt);
}

function parseOutputToJson(response: string) {
    try {
        return JSON.parse(response);
    } catch (error) {
        console.error("Error parsing output to JSON:", error);
        return null;
    }
}

// Main runner function
async function run() {
    const documents = await readPdfAsDocument('swiggy.pdf');
    const langChainModel = initializeLangChain();
    const response = await processDocument(documents, langChainModel);
    const jsonResponse = parseOutputToJson(response);
    console.log(jsonResponse);
}

run().catch(console.error);