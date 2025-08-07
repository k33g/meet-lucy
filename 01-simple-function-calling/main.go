package main

import (
	"context"
	"fmt"
	"os"
	"time"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
)

func GetToolsIndex() []openai.ChatCompletionToolParam {
	sayHelloTool := openai.ChatCompletionToolParam{
		Function: openai.FunctionDefinitionParam{
			Name:        "say_hello",
			Description: openai.String("Say hello to the given person name"),
			Parameters: openai.FunctionParameters{
				"type": "object",
				"properties": map[string]interface{}{
					"name": map[string]interface{}{
						"type": "string",
					},
				},
				"required": []string{"name"},
			},
		},
	}
	addTwoNumbersTool := openai.ChatCompletionToolParam{
		Function: openai.FunctionDefinitionParam{
			Name:        "add_two_numbers",
			Description: openai.String("Add two numbers together"),
			Parameters: openai.FunctionParameters{
				"type": "object",
				"properties": map[string]interface{}{
					"number1": map[string]interface{}{
						"type": "number",
					},
					"number2": map[string]interface{}{
						"type": "number",
					},
				},
				"required": []string{"number1", "number2"},
			},
		},
	}
	return []openai.ChatCompletionToolParam{
		sayHelloTool,
		addTwoNumbersTool,
	}
}

func main() {
	ctx := context.Background()

	// Docker Model Runner base URL
	baseURL := os.Getenv("MODEL_RUNNER_BASE_URL")

	type Model struct {
		Name  string
		Score int
	}

	model := Model{
		Name: os.Getenv("MODEL_LUCY_Q8_0"), Score: 0,
		//Name: os.Getenv("MODEL_QWEN3_LATEST"), Score: 0,
	}


	client := openai.NewClient(
		option.WithBaseURL(baseURL),
		option.WithAPIKey(""),
	)

	//userQuestion := openai.UserMessage("Say hello to Jean-Luc Picard")

	detectToolCall := func(model string, userQuestion string, theToolNameShouldBe string) (int, time.Duration) {

		success := 0

		params := openai.ChatCompletionNewParams{
			Messages: []openai.ChatCompletionMessageParamUnion{
				openai.UserMessage(userQuestion),
			},
			ParallelToolCalls: openai.Bool(false),
			Tools:             GetToolsIndex(),
			Model:             model,
			Temperature:       openai.Opt(0.0),
		}

		// Measure completion time
		startTime := time.Now()
		completion, err := client.Chat.Completions.New(ctx, params)
		duration := time.Since(startTime)
		if err != nil {
			fmt.Println("🔴 Model:", model, "Error:", err)
			return success, duration
		}

		toolCalls := completion.Choices[0].Message.ToolCalls

		// Return early if there are no tool calls
		if len(toolCalls) == 0 {
			if theToolNameShouldBe != "no_tool_call_expected" {
				fmt.Println("🔴 Model:", model, "No function call detected but expected:", theToolNameShouldBe)
			} else {
				fmt.Println("🟢 Model:", model, "No function call (not in the tools index)")
				success = 1
			}
			return success, duration
		}

		// Display the tool call(s)
		for _, toolCall := range toolCalls {

			if toolCall.Function.Name != theToolNameShouldBe {
				fmt.Println("🟡 Model:", model, "Function call detected:", toolCall.Function.Name, "but expected:", theToolNameShouldBe)
			} else {
				fmt.Println("🟢 Model:", model, "Function call detected:", toolCall.Function.Name, "with arguments:", toolCall.Function.Arguments)
				success = 1
			}
		}
		return success, duration

	}

	numberOfIterations := 10
	var totalDuration time.Duration

	for i := range numberOfIterations {
		fmt.Println(i, ". Running detection for models...")

		fmt.Println("🔵 Model:", model)
		userQuestion := "Tell me why the sky is blue and then say hello to Jean-Luc Picard. I love pineapple pizza!"
		success1, duration1 := detectToolCall(model.Name, userQuestion, "say_hello")

		userQuestion = "Where is Bob? Add 2 and 3. What is the capital of France?"
		success2, duration2 := detectToolCall(model.Name, userQuestion, "add_two_numbers")

		userQuestion = "The best pizza topping is pineapple. What is the capital of France? I love cooking."
		success3, duration3 := detectToolCall(model.Name, userQuestion, "no_tool_call_expected")

		iterationDuration := duration1 + duration2 + duration3
		totalDuration += iterationDuration

		model.Score += success1 + success2 + success3
		fmt.Printf("🟣 Model: %s Score: %d Duration for this iteration: %.2fs\n", model.Name, model.Score, iterationDuration.Seconds())

	}
	fmt.Println("Final scores:")

	fmt.Printf("- Model: %s Final Score: %d Percentage: %.1f%% Total Duration: %.2fs Average per completion: %.2fs\n",
		model.Name,
		model.Score,
		float64(model.Score)/float64(numberOfIterations*3)*100,
		totalDuration.Seconds(),
		totalDuration.Seconds()/float64(numberOfIterations*3))

	fmt.Println("Done!")

}
