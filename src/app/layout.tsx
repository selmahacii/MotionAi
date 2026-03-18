import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { Toaster } from "@/components/ui/toaster";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Selma Motion Pro - Precision Motion Analysis",
  description: "Advanced human motion analysis engine powered by Selma Motion Core. High-precision pose estimation and predictive modeling.",
  keywords: ["Motion Analysis", "Biomechanics", "Precision Engineering", "Selma Motion"],
  authors: [{ name: "Selma Haci" }],
  icons: {
    icon: "/favicon.ico",
  },
  openGraph: {
    title: "Selma Motion Pro",
    description: "Advanced human motion analysis engine",
    siteName: "Selma Motion",
    type: "website",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-background text-foreground`}
      >
        {children}
        <Toaster />
      </body>
    </html>
  );
}
