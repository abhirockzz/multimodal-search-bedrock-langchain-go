package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/url"
	"os"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/jackc/pgx/v5"
	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/vectorstores/pgvector"
)

var store pgvector.Store

const modelID = "amazon.titan-embed-image-v1"
const vectorDimension = 1024

func init() {

	host := "localhost"
	user := "postgres"
	password := "postgres"
	dbName := "postgres"

	connURLFormat := "postgres://%s:%s@%s:5432/%s?sslmode=disable"

	pgConnURL := fmt.Sprintf(connURLFormat, user, url.QueryEscape(password), host, dbName)

	embeddingModel, err := embeddings.NewEmbedder(embeddings.EmbedderClientFunc(CreateEmbedding))

	if err != nil {
		log.Fatal(err)
	}
	conn, err := pgx.Connect(context.Background(), pgConnURL)

	if err != nil {
		log.Fatal(err)
	}

	store, err = pgvector.New(
		context.Background(),
		//pgvector.WithPreDeleteCollection(true),
		pgvector.WithConn(conn),
		pgvector.WithEmbedder(embeddingModel),
		pgvector.WithVectorDimensions(vectorDimension),
	)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("vector store ready")
}

func main() {
	reader := bufio.NewReader(os.Stdin)

	numOfResults := 5

	for {
		fmt.Print("\nEnter your message: ")
		question, _ := reader.ReadString('\n')
		question = strings.TrimSpace(question)

		results, err := store.SimilaritySearch(context.Background(), question, numOfResults)

		if err != nil {
			log.Fatal(err)
		}

		fmt.Println("=====RESULTS=======")

		for _, result := range results {
			fmt.Println("[search result with score]:", result.Metadata["source"], (1 - result.Score)) // subtract by one since for cosine similarity
		}
		fmt.Println("============")

	}
}

type Request struct {
	InputText       string          `json:"inputText"`
	EmbeddingConfig EmbeddingConfig `json:"embeddingConfig"`
}

type EmbeddingConfig struct {
	OutputEmbeddingLength int `json:"outputEmbeddingLength"`
}

type Response struct {
	Embedding []float32 `json:"embedding"`
}

func CreateEmbedding(ctx context.Context, texts []string) ([][]float32, error) {

	req := Request{
		InputText: texts[0],
		EmbeddingConfig: EmbeddingConfig{
			OutputEmbeddingLength: vectorDimension,
		},
	}

	reqJSON, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	cfg, err := config.LoadDefaultConfig(context.Background())
	if err != nil {
		log.Fatal(err)
	}

	brc := bedrockruntime.NewFromConfig(cfg)

	result, err := brc.InvokeModel(context.Background(), &bedrockruntime.InvokeModelInput{
		ModelId:     aws.String(modelID),
		Accept:      aws.String("*/*"),
		ContentType: aws.String("application/json"),
		Body:        reqJSON,
	})
	if err != nil {
		return nil, err
	}

	var response Response
	err = json.Unmarshal(result.Body, &response)
	if err != nil {
		return nil, err
	}

	return [][]float32{response.Embedding}, nil
}
