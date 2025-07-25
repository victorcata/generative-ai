import { OpenAIClient, AzureKeyCredential } from "@azure/openai";
import * as dotenv from "dotenv";

dotenv.config();

const endpoint = process.env.AZURE_FOUNDRY_ENDPOINT || "";
const azureApiKey = process.env.AZURE_FOUNDRY_API_KEY || "";
const deployment = process.env.AZURE_FOUNDRY_GPT_DEPLOYMENT || "";

export async function main() {
  try {
    console.log("== Chat Completions App ==");

    const client = new OpenAIClient(
      endpoint,
      new AzureKeyCredential(azureApiKey)
    );
    const deploymentName = deployment;

    const result = await client.getChatCompletions(
      deploymentName,
      [
        { role: "system", content: "You're the president of France" },
        { role: "system", content: "You have just resigned" },
        { role: "user", content: "What tasks needs doing?" },
      ],
      {
        maxTokens: 100,
      }
    );

    for (const choice of result.choices) {
      console.log(choice.message);
    }
  } catch (error) {
    console.log("The sample encoutered an error: ", error);
  }
}

main();
