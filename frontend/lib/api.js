const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/jpg"];
const ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png"];

export function validateImageFile(file) {
  if (!file) {
    return "Please select an image before predicting.";
  }

  const extension = file.name.includes(".")
    ? `.${file.name.split(".").pop().toLowerCase()}`
    : "";

  if (!ALLOWED_EXTENSIONS.includes(extension)) {
    return "Unsupported file type. Please upload a JPG, JPEG, or PNG image.";
  }

  if (file.type && !ALLOWED_TYPES.includes(file.type)) {
    return "Unsupported file type. Please upload a JPG, JPEG, or PNG image.";
  }

  return null;
}

async function parseErrorMessage(response) {
  try {
    const error = await response.json();

    if (typeof error.detail === "string") {
      return error.detail;
    }

    if (Array.isArray(error.detail)) {
      return error.detail.map((item) => item.msg).join(", ");
    }
  } catch {
    // Use default message below.
  }

  return "Prediction request failed. Please try again.";
}

export async function predictImage(file) {
  const validationError = validateImageFile(file);
  if (validationError) {
    throw new Error(validationError);
  }

  const formData = new FormData();
  formData.append("file", file);

  let response;

  try {
    response = await fetch(`${API_URL}/predict`, {
      method: "POST",
      body: formData,
    });
  } catch {
    throw new Error(
      "Cannot reach the backend server. Make sure it is running on port 8000."
    );
  }

  if (!response.ok) {
    throw new Error(await parseErrorMessage(response));
  }

  const data = await response.json();

  const probabilities = data.probabilities || data.class_probabilities;

  if (
    !data ||
    typeof data.predicted_class !== "string" ||
    typeof data.confidence !== "number" ||
    !probabilities
  ) {
    throw new Error("Unexpected response format from the server.");
  }

  return {
    ...data,
    probabilities,
  };
}
