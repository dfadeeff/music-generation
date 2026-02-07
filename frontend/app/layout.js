import "./globals.css";

export const metadata = {
  title: "Music to My Ears",
  description: "AI Music Generation powered by MusicGen",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body style={{ margin: 0, fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif' }}>
        {children}
      </body>
    </html>
  );
}