package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/openai/openai-go/shared/constant"
)

// FunctionCallRecord tracks each function call and its result
type FunctionCallRecord struct {
	FunctionName string
	Arguments    string
	Result       string
	CallID       string
	Duration     time.Duration
}

func GetToolsIndex() []openai.ChatCompletionToolParam {
	calculateSumTool := openai.ChatCompletionToolParam{
		Function: openai.FunctionDefinitionParam{
			Name:        "calculate_sum",
			Description: openai.String("Calculate the sum of two numbers"),
			Parameters: openai.FunctionParameters{
				"type": "object",
				"properties": map[string]interface{}{
					"a": map[string]string{
						"type":        "number",
						"description": "The first number",
					},
					"b": map[string]string{
						"type":        "number",
						"description": "The second number",
					},
				},
				"required": []string{"a", "b"},
			},
		},
	}

	sayHelloTool := openai.ChatCompletionToolParam{
		Function: openai.FunctionDefinitionParam{
			Name:        "say_hello",
			Description: openai.String("Say hello to the given name"),
			Parameters: openai.FunctionParameters{
				"type": "object",
				"properties": map[string]interface{}{
					"name": map[string]string{
						"type":        "string",
						"description": "The name to greet",
					},
				},
				"required": []string{"name"},
			},
		},
	}

	return []openai.ChatCompletionToolParam{
		calculateSumTool,
		sayHelloTool,
	}
}

func executeFunction(functionName string, arguments string) string {
	switch functionName {
	case "say_hello":
		var args struct {
			Name string `json:"name"`
		}
		if err := json.Unmarshal([]byte(arguments), &args); err != nil {
			return `{"error": "Invalid arguments for say_hello"}`
		}
		hello := fmt.Sprintf("👋 Hello, %s!🙂", args.Name)
		return fmt.Sprintf(`{"message": "%s"}`, hello)

	case "calculate_sum":
		var args struct {
			A float64 `json:"a"`
			B float64 `json:"b"`
		}
		if err := json.Unmarshal([]byte(arguments), &args); err != nil {
			return `{"error": "Invalid arguments for calculate_sum"}`
		}
		sum := args.A + args.B
		return fmt.Sprintf(`{"result": %g}`, sum)

	default:
		return `{"error": "Unknown function"}`
	}
}

func main() {
	ctx := context.Background()

	baseURL := os.Getenv("MODEL_RUNNER_BASE_URL")

	modelName := os.Getenv("MODEL_LUCY_Q8_0")
	//modelName := os.Getenv("MODEL_QWEN3_LATEST")

	client := openai.NewClient(
		option.WithBaseURL(baseURL),
		option.WithAPIKey(""),
	)

	// userMessage := `/think
	// 	Make the sum of 40 and 2, 
	// 	then say hello to Bob and to Sam, 
	// 	make the sum of 5 and 37
	// 	Say hello to Alice
	// `

	// userMessage := `
	// 	Make the sum of 30 and 2,
	// 	If the result is higher than 40
	// 	Then say hello to Bob Else to Sam
	// `

	userMessage := `
		Make the sum of 40 and 2,
		If the result is higher than 40
		Then say hello to Bob Else to Sam
	`


	messages := []openai.ChatCompletionMessageParamUnion{
		openai.UserMessage(userMessage),
	}

	// Track all function calls and results
	var functionCallHistory []FunctionCallRecord
	stopped := false

	for !stopped {
		fmt.Println("⏳ Making function call request...")
		
		startTime := time.Now()

		params := openai.ChatCompletionNewParams{
			Messages:    messages,
			Tools:       GetToolsIndex(),
			Model:       modelName,
			Temperature: openai.Opt(0.0),
			ToolChoice: openai.ChatCompletionToolChoiceOptionUnionParam{
				OfAuto: openai.String("auto"),
			},
		}

		completion, err := client.Chat.Completions.New(ctx, params)
		if err != nil {
			fmt.Printf("🔴 Error: %v\n", err)
			return
		}
		
		completionDuration := time.Since(startTime)

		finishReason := completion.Choices[0].FinishReason

		switch finishReason {
		case "tool_calls":
			detectedToolCalls := completion.Choices[0].Message.ToolCalls

			if len(detectedToolCalls) > 0 {
				// Add assistant message with tool calls to conversation history FIRST
				// This is critical for proper conversation flow in the LLM
				toolCallParams := make([]openai.ChatCompletionMessageToolCallParam, len(detectedToolCalls))
				for i, toolCall := range detectedToolCalls {

					toolCallParams[i] = openai.ChatCompletionMessageToolCallParam{
						ID:   toolCall.ID,
						Type: constant.Function("function"),
						Function: openai.ChatCompletionMessageToolCallFunctionParam{
							Name:      toolCall.Function.Name,
							Arguments: toolCall.Function.Arguments,
						},
					}
				}

				// Create assistant message with tool calls using proper union type
				assistantMessageParam := openai.ChatCompletionAssistantMessageParam{
					ToolCalls: toolCallParams,
				}
				assistantMessage := openai.ChatCompletionMessageParamUnion{
					OfAssistant: &assistantMessageParam,
				}

				// Add the assistant message with tool calls to the conversation history
				messages = append(messages, assistantMessage)


				fmt.Println("")
				fmt.Println("🚀 Processing tool calls...")

				for _, toolCall := range detectedToolCalls {
					functionName := toolCall.Function.Name
					functionArgs := toolCall.Function.Arguments
					callID := toolCall.ID

					fmt.Printf("▶️ Executing function: %s with args: %s\n", functionName, functionArgs)

					resultContent := executeFunction(functionName, functionArgs)

					fmt.Printf("Function result: %s with CallID: %s\n\n", resultContent, callID)

					// Add the tool call result to the conversation history
					messages = append(
						messages,
						openai.ToolMessage(
							resultContent,
							toolCall.ID,
						),
					)

					// Record this function call for the summary (no impact on the conversation)
					functionCallHistory = append(functionCallHistory, FunctionCallRecord{
						FunctionName: functionName,
						Arguments:    functionArgs,
						Result:       resultContent,
						CallID:       callID,
						Duration:     completionDuration,
					})
				}

			} else {
				fmt.Println("😢 No tool calls found in response")
			}

		case "stop":
			fmt.Println("🟥 Stopping due to 'stop' finish reason.")
			stopped = true
			assistantMessage := completion.Choices[0].Message.Content
			fmt.Printf("🤖 %s\n", assistantMessage)

			// Add final assistant message to conversation history
			messages = append(messages, openai.AssistantMessage(assistantMessage))

		default:
			fmt.Printf("🔴 Unexpected response: %s\n", finishReason)
			stopped = true
		}

	}

	// Display summary of all function calls
	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("📋 FUNCTION CALL SUMMARY")
	fmt.Println(strings.Repeat("=", 60))

	if len(functionCallHistory) == 0 {
		fmt.Println("❌ No function calls were executed")
	} else {
		fmt.Printf("✅ Total function calls executed: %d\n\n", len(functionCallHistory))

		for i, call := range functionCallHistory {
			fmt.Printf("%d. Function: %s\n", i+1, call.FunctionName)
			fmt.Printf("   Arguments: %s\n", call.Arguments)
			fmt.Printf("   Result: %s\n", call.Result)
			fmt.Printf("   Call ID: %s\n", call.CallID)
			fmt.Printf("   Duration: %v\n", call.Duration)
			if i < len(functionCallHistory)-1 {
				fmt.Println()
			}
		}
	}

	fmt.Println(strings.Repeat("=", 60))
}
