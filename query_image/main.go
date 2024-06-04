package main

import (
	"bufio"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
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
		fmt.Print("\nEnter image source: ")
		imagePath, _ := reader.ReadString('\n')
		imagePath = strings.TrimSpace(imagePath)

		base64EncodedImage, err := imageToBase64String(imagePath)
		if err != nil {
			log.Fatal(err)
		}

		results, err := store.SimilaritySearch(context.Background(), base64EncodedImage, numOfResults)

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
	InputImage      string          `json:"inputImage"`
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
		InputImage: texts[0],
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

func imageToBase64String(source string) (string, error) {

	var imageBytes []byte

	if strings.Contains(source, "http") {
		resp, err := http.Get(source)
		if err != nil {
			return "", err
		}
		defer resp.Body.Close()

		imageBytes, err = io.ReadAll(resp.Body)
		if err != nil {
			return "", err
		}
	} else {
		//assume it's local
		var err error
		imageBytes, err = os.ReadFile(source)
		if err != nil {
			return "", err
		}
	}

	encodedString := base64.StdEncoding.EncodeToString(imageBytes)

	return encodedString, nil
}
