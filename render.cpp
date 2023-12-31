#include "render.h"

renderWindow::renderWindow() 
    : window(sf::VideoMode(1100, 1100), "again", sf::Style::Close)
{
    font.loadFromFile("Resources/Fonts/arial.ttf");

    titleStr.setFont(font);

    lossStr.setFont(font);
    lossStr.setString("0");
    lossStr.setPosition(0,40);

    gradientStr.setFont(font);

    rect.setSize(sf::Vector2f(8.0f, 8.0f));
    rect.setFillColor(sf::Color::White);

    imageTexture.create(32,32);
    imageSprite.setPosition(40,160);
    pixels = new sf::Uint8[32 * 32 * 4];

}

void renderWindow::BeginDisplay()
{
    window.clear(sf::Color::Black);
}

void renderWindow::DisplayTitle(int epoch, double loss, const char* text)
{
    titleStr.setString(text);
    window.draw(titleStr);

    char buf[200];
    snprintf(buf, sizeof(buf),  "Epoch: %d loss: %f", epoch, loss);
    lossStr.setString(buf);
    window.draw(lossStr);
}

void renderWindow::DisplayImage(
    unsigned char* r,
    unsigned char* g,
    unsigned char* b,
    int x, int y)
{
    for (int i=0; i < 1024; i++)
    {
        int o = i*4;
        pixels[o+0] = r[i];
        pixels[o+1] = g[i];
        pixels[o+2] = b[i];
        pixels[o+3] = 255;
    }
    imageTexture.update(pixels);
    imageSprite.setTexture(imageTexture);
    imageSprite.setPosition(x,y);
    window.draw(imageSprite);
}

void renderWindow::DisplayGrid(
    const column& gradients,
    const column& activations1,
    const column& activations2,
    const column& activations3,    
    const int gridSize,
    matrix& values)
{
    float tpos = 840.0f;
    float tdiff = 50.f;
    for (int g=0; g< gradients.size(); g++)
    {
        char gbuf[200];
        snprintf(gbuf, sizeof(gbuf), "delta[%d]: %f", g, gradients[g]);
        gradientStr.setString(gbuf);
        gradientStr.setPosition(tpos, 20 + g*tdiff);
        window.draw(gradientStr);
    }

    for (int g=0; g< gradients.size(); g++)
    {
        char gbuf[200];
        snprintf(gbuf, sizeof(gbuf), "val[%d]: %f", g, activations1[g]);
        gradientStr.setString(gbuf);
        gradientStr.setPosition(tpos, 320 + g*tdiff);
        window.draw(gradientStr);
    }

    for (int g=0; g< gradients.size(); g++)
    {
        char gbuf[200];
        snprintf(gbuf, sizeof(gbuf), "val[%d]: %f", g, activations2[g]);
        gradientStr.setString(gbuf);
        gradientStr.setPosition(tpos, 520 + g*tdiff);
        window.draw(gradientStr);
    }

    for (int g=0; g< gradients.size(); g++)
    {
        char gbuf[200];
        snprintf(gbuf, sizeof(gbuf), "val[%d]: %f", g, activations3[g]);
        gradientStr.setString(gbuf);
        gradientStr.setPosition(tpos, 720 + g*tdiff);
        window.draw(gradientStr);
    }

    for (int y=0; y < gridSize; y++)
    {
        for (int x=0; x < gridSize; x++)
        {
            const int i = y * gridSize + x;
            const column& rgb = values[i];

            sf::Uint8 r = sf::Uint8(rgb[0] * 255);
            sf::Uint8 g = sf::Uint8(rgb[1] * 255);
            sf::Uint8 b = sf::Uint8(rgb[2] * 255);

            rect.setPosition(sf::Vector2f(20 + x*10*1.0f, 100 + y*10*1.0f));
            rect.setFillColor(sf::Color(r, g, b,  255));

            window.draw(rect);
        }
    }
}

void renderWindow::EndDisplay()
{
    window.display();
}

void renderWindow::ProcessEvents(bool& running)
{
    sf::Event event;
    while (window.pollEvent(event))
    {
        switch (event.type)
        {
            case sf::Event::Closed:
            {
                window.close();
                running = false;
                break;
            }
            case sf::Event::KeyReleased:
            {
                switch (event.key.code)
                {
                    case sf::Keyboard::Escape:
                    {
                        window.close();
                        running = false;
                        break;
                    }
                }
            }
        }
    }
}
