package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log"
	"net/url"
	"os"
	"path/filepath"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/jackc/pgx/v5"
	"github.com/tmc/langchaingo/embeddings"
	"github.com/tmc/langchaingo/schema"
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

	root := "sample_images"

	err := filepath.Walk(root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() {
			load(path)
		}
		return nil
	})

	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("data successfully loaded into vector store")
}

func load(source string) {

	doc, err := imageToLangchainDoc(source)

	if err != nil {
		log.Fatal(err)
	}

	_, err = store.AddDocuments(context.Background(), []schema.Document{doc})

	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("successfully loaded", source, "into vector store")
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

func imageToLangchainDoc(source string) (schema.Document, error) {

	imageBytes, err := os.ReadFile(source)
	if err != nil {
		return schema.Document{}, err
	}

	encodedString := base64.StdEncoding.EncodeToString(imageBytes)

	return schema.Document{PageContent: encodedString, Metadata: map[string]any{"source": source}}, nil
}
