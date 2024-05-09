package org.example.server;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import fi.iki.elonen.NanoHTTPD;


import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

public class MultiLayerNetworkView extends NanoHTTPD {
    public static String DATA_NETWORK = "";

    public MultiLayerNetworkView(int port) {
        super(port);
    }

    @Override
    public Response serve(IHTTPSession session) {
        Method method = session.getMethod();
        if (Method.GET.equals(method)) {
            return handleGetRequest(session);
        } else if (Method.POST.equals(method)) {
            return handlePostRequest(session);
        } else {
            return newFixedLengthResponse(Response.Status.NOT_FOUND, MIME_PLAINTEXT, "Not Found");
        }
    }

    private Response handleGetRequest(IHTTPSession session) {
        // Вернуть информацию о нейронной сети
        return newFixedLengthResponse(DATA_NETWORK);
    }

    private Response handlePostRequest(IHTTPSession session) {
        // Обучить нейронную сеть на основе данных из запроса
        Map<String, String> files = new HashMap<>();
        try {
            session.parseBody(files);
        } catch (IOException | ResponseException e) {
            e.printStackTrace();
            return newFixedLengthResponse(Response.Status.INTERNAL_ERROR, MIME_PLAINTEXT, "Internal Server Error");
        }
        String postData = files.get("postData");
        // Преобразовать данные в формат, подходящий для обучения нейронной сети
        TrainingData trainingData = parseTrainingData(postData);

        // Обучить нейронную сеть на основе данных из запроса
        trainNetwork(trainingData);

        return newFixedLengthResponse(getText());
    }

    private TrainingData parseTrainingData(String postData) {
        ObjectMapper objectMapper = new ObjectMapper();
        TrainingData trainingData = null;
        try {
            trainingData = objectMapper.readValue(postData, TrainingData.class);
        } catch (JsonProcessingException e) {
            e.printStackTrace();
        }
        return trainingData;
    }

    private void trainNetwork(TrainingData trainingData) {
        // Здесь добавьте логику обучения нейронной сети на основе trainingData
        // ...
    }

    public String getText() {
        String msg = "<!DOCTYPE html>\n" +
                "<html lang=\"en\">\n" +
                "<head>\n" +
                "    <meta charset=\"UTF-8\">\n" +
                "    <title>OpenBox</title>\n" +
                "</head>\n" +
                "<body>\n" +
                "    <canvas id=\"myCanvas\" width=\"3000\" height=\"500\"></canvas>\n" +
                "    <script type=\"application/javascript\" language=\"javascript\">\n" +
                "        // JavaScript-код здесь...\n" +
                "    </script>\n" +
                "</body>\n" +
                "</html>";
        return msg;
    }
}
