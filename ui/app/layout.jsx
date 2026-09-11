import SiteFooter from "@/components/SiteFooter";
import SiteHeader from "@/components/SiteHeader";
import "./globals.css";

export const metadata = {
  title: {
    default: "RAVIN",
    template: "%s | RAVIN",
  },
  description:
    "RAVIN helps La Trobe University students and staff find clear, source-supported policy information.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body>
        <a className="skip-link" href="#main-content">
          Skip to main content
        </a>
        <SiteHeader />
        {children}
        <SiteFooter />
      </body>
    </html>
  );
}
