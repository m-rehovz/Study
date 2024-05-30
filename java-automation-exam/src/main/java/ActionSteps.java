import com.codeborne.selenide.SelenideElement;

import static com.codeborne.selenide.Condition.text;
import static com.codeborne.selenide.Condition.visible;
import static com.codeborne.selenide.Selenide.$x;

public class ActionSteps {

    public void setValueInField(SelenideElement element, String value) {
        element.setValue(value);
    }

    public void clickOnWall(SelenideElement element) {
        element.click();
    }

    public void selectCheckbox(SelenideElement element) {
        element.shouldBe(visible);
        element.click();
        element.isSelected();
    }

    public void clickBtn(SelenideElement element) {
        element.click();
    }

    public void checkText(SelenideElement element, String expectedText) {
        element.shouldHave(text(expectedText));
    }

}
