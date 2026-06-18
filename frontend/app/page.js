import ClassifierApp from "@/components/ClassifierApp";

export default function HomePage() {
  return (
    <main className="min-h-screen bg-slate-50 px-4 py-10">
      <div className="mx-auto flex w-full max-w-3xl flex-col gap-6">
        <section className="space-y-2 text-center">
          <h1 className="text-3xl font-semibold tracking-tight text-slate-900">
            Medical Image Classifier
          </h1>
          <p className="text-sm text-slate-500 sm:text-base">
            Upload a medical image to get an educational prediction from the AI
            model.
          </p>
        </section>

        <ClassifierApp />
      </div>
    </main>
  );
}
